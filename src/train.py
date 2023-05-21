import torch
import math
import numpy as np
from torch.utils.data import DataLoader
import math
from transformers import  get_cosine_schedule_with_warmup
from tqdm import tqdm
import gc
import open_clip
from misc import set_seed, AverageMeter
from dataset import Swan_dataset
from model import Model, ArcFace_criterion, get_lr_groups
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from misc import get_augs
from config import CFG
import json


def train(model, train_loader, optimizer, scaler, scheduler, epoch):
    model.train()
    loss_metrics = AverageMeter()
    criterion = ArcFace_criterion

    tmp = np.sqrt(1 / np.sqrt(value_counts))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * CFG.m + CFG.m_min
        
    bar = tqdm(train_loader, disable=False)
    for step, data in enumerate(bar):
        step += 1
        images = data[0].to(CFG.device, dtype=torch.float)
        labels = data[1].to(CFG.device)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=CFG.autocast):
            outputs, _ = model(images)

        loss = criterion(outputs, labels, margins, CFG.s, CFG.n_classes)
        loss_metrics.update(loss.item(), batch_size)
        loss = loss / CFG.acc_steps
        scaler.scale(loss).backward()

        if step % CFG.acc_steps == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            CFG.global_step += 1
            
        lrs = get_lr_groups(optimizer.param_groups)
        loss_avg = loss_metrics.avg
        bar.set_postfix(loss=loss_avg, epoch=epoch, lrs=lrs, step=CFG.global_step)


def validate(model, loader):
    model.eval()
    preds = []
    labels = []
    for images, label in loader:
        with torch.no_grad():
            outputs = model(images.to(CFG.device))[0].argmax(1).cpu().detach().tolist()
            preds.extend(outputs)
            labels.extend(label)
            
    metric = f1_score(labels, preds, average='macro')
    return metric



if __name__=='__main__':
    set_seed(CFG.seed)

    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms(CFG.model_name, pretrained=CFG.model_data)
    image_size = model_transforms.transforms[0].size[0]
    mean, std = model_transforms.transforms[-1].mean, model_transforms.transforms[-1].std
    
    with open('/media/ivan/Data1/hack_ai/train_dataset_Минприроды/boxes.json') as f:
        boxes = json.load(f)


    train_dataset = Swan_dataset(
            CFG.df,
            boxes=boxes,
            mode='train', 
            transform=get_augs(mean=mean, std=std, image_size=image_size, mode='train'),
        )
    valid_dataset = Swan_dataset(
            CFG.df,
            boxes=boxes,
            mode='valid', 
            transform=get_augs(mean=mean, std=std, image_size=image_size, mode='valid'),
        )

    train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=CFG.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, num_workers=8, batch_size=CFG.valid_batch_size, shuffle=False)


    value_counts = CFG.df['label_idx'].value_counts().tolist()
    model = Model(vit_backbone.cpu(), cfg=CFG).to(CFG.device)
    optimizer = torch.optim.AdamW(model.get_parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    steps_per_epoch = math.ceil(len(train_dataloader) / CFG.acc_steps)
    num_training_steps = math.ceil(CFG.n_epochs * steps_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_training_steps=num_training_steps,
                                                num_warmup_steps=CFG.n_warmup_steps)   
    CFG.global_step = 0                   
    for epoch in range(math.ceil(CFG.n_epochs)):
        train(model, train_dataloader, optimizer, scaler, scheduler, epoch)
        score = validate(model, valid_loader)
        print(f'Epoch = {epoch}, score: {score}')
        torch.save(model.state_dict(), f'checkpoints/{CFG.model_name}_{CFG.model_data}_{score}.pth')
        gc.collect()
        torch.cuda.empty_cache()



