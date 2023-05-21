from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import glob
import os
import albumentations as A
import open_clip
from model import Head
import pandas as pd
import argparse
from torch import nn
import glob

parser = argparse.ArgumentParser(description='Test data')
parser.add_argument('--path', help='path to test data')
parser.add_argument('--ckpt', help='path to ckpt')
args = parser.parse_args()



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



class CFG:
    mapping = {0:3, 1:2, 2:1}
    device = 'cuda'
    emb_size = 512
    n_classes = 3
    ckpt = ''

def test_model(path, ckpt, submission_name):
    model_name = os.path.basename(ckpt).split('_')[0]
    score = os.path.basename(ckpt).split('_')[-1]
    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms(model_name, pretrained=False)
    image_size = model_transforms.transforms[0].size[0]
    mean, std = model_transforms.transforms[-1].mean, model_transforms.transforms[-1].std

    model = Model(vit_backbone, CFG).to(CFG.device)

    if ckpt:
        model.load_state_dict(torch.load(ckpt))
        model.eval()

    augs = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std, p=1)
            ])
    test_dataset = Swan_dataset_test(path, augs)
    test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=8)
    
    final_preds, final_paths = [], []
    for batch in test_dataloader:
        images, paths = batch
        images = images.to(CFG.device)

        with torch.no_grad():
            pred = model(images).detach().cpu().argmax(1).tolist()
            final_preds.extend(pred)
            final_paths.extend(paths)

    result_csv = pd.DataFrame(columns=['name', 'class'])
    result_csv.loc[:, 'name'] = final_paths
    result_csv.loc[:, 'class'] = final_preds
    result_csv['class'] = result_csv['class'].map(CFG.mapping)

    result_csv.to_csv(submission_name+f'_local={score}.csv', sep=';', index=False)

if __name__=='__main__':
    
    path = args.path
    # ckpt = args.ckpt
    for idx, ckpt in enumerate(glob.glob('/media/ivan/Data1/hack_ai/train_dataset_Минприроды/*pth')):
        print(ckpt)
        test_model(path, ckpt, f'submission_{idx}')


