import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

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
    
    
class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
            
    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     

def ArcFace_criterion(logits_m, target, margins, s, n_classes):
    arc = ArcFaceLossAdaptiveMargin(margins=margins, s=s)
    loss_m = arc(logits_m, target, n_classes)
    return loss_m


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
        return self.head(x)

    def get_parameters(self):

        parameter_settings = [] 
        parameter_settings.extend(self.get_parameter_section([(n, p) for n, p in self.vit_backbone.named_parameters()], lr=self.cfg.vit_bb_lr, wd=self.cfg.vit_bb_wd)) 

        parameter_settings.extend(self.get_parameter_section([(n, p) for n, p in self.head.named_parameters()], lr=self.cfg.hd_lr, wd=self.cfg.hd_wd)) 

        return parameter_settings

    def get_parameter_section(self, parameters, lr=None, wd=None): 
        parameter_settings = []

        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for no, (n,p) in enumerate(parameters):
            for split in n.split('.'):
                if split.isnumeric():
                    layer_no = int(split)
            
            if not layer_no:
                layer_no = 0
            if lr_is_dict:
                for k,v in lr.items():
                    if layer_no < int(k):
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k,v in wd.items():
                    if layer_no < int(k):
                        temp_wd = v
                        break
            else:
                temp_wd = wd

            parameter_setting = {"params" : p, "lr" : temp_lr, "weight_decay" : temp_wd}
            parameter_settings.append(parameter_setting)

        return parameter_settings
    
def get_lr_groups(param_groups):
        groups = sorted(set([param_g['lr'] for param_g in param_groups]))
        groups = ["{:2e}".format(group) for group in groups]
        return groups