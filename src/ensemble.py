import torch
import math
from torch import nn
import open_clip
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import glob
import os
from tqdm import tqdm
import glob
import numpy as np
from PIL import Image
import torch.nn.functional as F
import pandas as pd
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

class CFG:
    emb_size = 512
    n_classes = 3

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   

class Multisample_Dropout(nn.Module):
    def __init__(self):
        super(Multisample_Dropout, self).__init__()
        self.dropout = nn.Dropout(.1)
        self.dropouts = nn.ModuleList([nn.Dropout((i+1)*.1) for i in range(5)])
        
    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts],dim=0),dim=0) 



class Swan_dataset_test(Dataset):
    def __init__(self, dirpath, augs):
        self.fnames = glob.glob(os.path.join(dirpath, '*'))
        self.augs = augs

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        path = self.fnames[idx]
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        if self.augs:
            img = self.augs(image=img)['image']
            img = torch.from_numpy(img).permute(2, 0, 1)
    
        return img, os.path.basename(path)
    
class Head(nn.Module):
    def __init__(self, hidden_size, emb_size, n_classes):
        super(Head, self).__init__()

        self.emb = nn.Linear(hidden_size, emb_size, bias=False)
        self.arc = ArcMarginProduct_subcenter(emb_size, n_classes)
        self.dropout = Multisample_Dropout()

    def forward(self, x):
        embeddings = self.dropout(x, self.emb)
        
        output = self.arc(embeddings)

        return output, F.normalize(embeddings)
    
class Model(nn.Module):
    def __init__(self, vit_backbone, cfg):
        super(Model, self).__init__()
        
        self.cfg = cfg
        vit_backbone = vit_backbone.visual
        self.img_size = vit_backbone.image_size
        if type(self.img_size)==tuple:
            self.img_size = self.img_size[1]
        hidden_size = vit_backbone(torch.zeros((1, 3, self.img_size, self.img_size))).shape[1]
        self.vit_backbone = vit_backbone
        self.head = Head(hidden_size, self.cfg.emb_size, self.cfg.n_classes)

    def forward(self, x):

        x = self.vit_backbone(x)
        return self.head(x)[0]



vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained=False)
image_size = model_transforms.transforms[0].size[0]
mean, std = model_transforms.transforms[-1].mean, model_transforms.transforms[-1].std
val_aug = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std, p=1)
    ])

valid_ds = Swan_dataset_test(
    '/media/ivan/Data1/hack_ai/Тестовый датасет/', 
    augs=val_aug
)

dl = DataLoader(valid_ds, shuffle=False, batch_size=128, num_workers=8)
classif_model = Model(vit_backbone.cpu(), CFG).to('cpu')
classif_model.cuda()
classif_model.eval()



path2pred = {key: np.array([0., 0., 0.]) for key in valid_ds.fnames}
# path2pred = {key: [] for key in valid_ds.df['img_path']}


model_names = [
 'checkpoints/ViT-L-14-336_openai_1_0.9925378207536739.pth',
 'checkpoints/ViT-L-14-336_openai_1_0.994579411079596.pth',
 'checkpoints/ViT-L-14-336_openai_1_0.9945824842787966.pth',
 'checkpoints/ViT-L-14-336_openai_2_0.9921492526661186.pth',
 'checkpoints/ViT-L-14-336_openai_2_0.995448866536195.pth',
 'checkpoints/ViT-L-14-336_openai_3_0.9932698668618455.pth',
 'checkpoints/ViT-L-14-336_openai_2_0.992174262897354.pth'
 ]


for model_name in model_names:
    classif_model.load_state_dict(
    torch.load(model_name, 
               map_location='cuda'), 
    
    )
    classif_model.cuda()
    classif_model.eval()
    
    preds = []
    for batch in tqdm(dl):
        with torch.no_grad():
            pred = classif_model(batch[0].cuda())
            preds.extend(pred.cpu().detach().softmax(1).numpy())

    for idx, i in enumerate(preds):
        path2pred[valid_ds.fnames[idx]]+= i        



model = YOLO('checkpoints/best.pt')

path2boxes = {}

for path in valid_ds.fnames:
    result = model.predict(path)
    path2boxes[path] = result[0].boxes.xyxy.cpu().detach()

classif_model.load_state_dict(
    torch.load('checkpoints/ViT-L-14-336_openai_3_0.9932698668618455.pth', 
               map_location='cuda'), 
    )

path2pred = {}
for path in tqdm(path2boxes):
    path2pred[path] = np.zeros(3, dtype=float)
    img = Image.open(path).convert('RGB')
    w, h = img.size
    img = np.array(img)
    boxes = path2boxes[path].numpy()
    crops = []
    for box in boxes:
        box = [int(i) for i in box]
        x1, y1, x2, y2 = box
        h_box = y2-y1
        w_box = x2-x1
        
        h_box = int(h_box*1.2)
        w_box = int(w_box*1.2)
        
        x1 = max(0, x1-(w_box//2))
        x2 = min(w, x2+(w_box//2))
        y1 = max(0, y1-(h_box//2))
        y2 = min(h, y2+(h_box//2))
        
        crop = img[y1:y2, x1:x2]
        crop = val_aug(image=crop)['image']
        crop = torch.from_numpy(crop).permute(2, 0, 1)
        crops.append(crop)
    batch = torch.stack(crops).cuda()
    with torch.no_grad():
        predict = classif_model(batch).cpu().detach().softmax(1).numpy()
        for i in predict:
            path2pred[path]+=i

sub = pd.DataFrame(columns=['name', 'class'])

mapping = {0:3, 1:2, 2:1}
for path in path2pred:
    sub.loc[len(sub)] = [os.path.basename(path), mapping[path2pred[path].argmax()]]

sub.to_csv('/media/ivan/Data1/hack_ai/src/ensemble.csv', index=False, sep=';')