import torch
import numpy as np
import os
import albumentations as A



def set_seed(seed):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_augs(mean, std, image_size, mode='train'):
    if mode=='train':
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ImageCompression(quality_lower=99, quality_upper=100),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
                A.Resize(image_size, image_size),
                A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
                A.OneOf([
                    A.ChannelShuffle(),
                    A.ChannelDropout(),
                    A.ColorJitter(),
                    A.ToGray(),
                ], p=0.65),
                A.Normalize(mean=mean, std=std, p=1), 
            ])

    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std, p=1)
        ])