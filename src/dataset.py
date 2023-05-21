from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class Swan_dataset(Dataset):
    def __init__(self, 
                 df, 
                 boxes, 
                 mode='train', 
                 transform=None
                 ):
        self.df = df[df['split']==mode].reset_index(drop=True)
        self.boxes = boxes
        self.augs = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.loc[idx, 'img_path']
        label = self.df.loc[idx, 'label_idx']
                    
        try:
            # trying to take a swan only from whole image
            use_box = np.random.random()>1
            if use_box and self.boxes.get(path):
                boxes = np.array(self.boxes[path]['boxes'])
                confs = np.array(self.boxes[path]['confs'])
                good_boxes = boxes[confs>0.8]
                if len(good_boxes):
                    box_idx = np.random.randint(0, len(good_boxes))
                    box = boxes[box_idx]
                else:
                    use_box = False
            else:
                use_box = False
            

            img = Image.open(path).convert('RGB')
            img = np.array(img)
            
            if use_box:
                box = [int(k) for k in box]
                x1, y1, x2, y2 = box
                img = img[y1:y2, x1:x2]

            if self.augs:
                img = self.augs(image=img)['image']
                img = torch.from_numpy(img).permute(2, 0, 1)
        except Exception as e:
            print(e, path)
            img = torch.zeros((3, 320, 320))
        
        return img, label