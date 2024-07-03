import pandas as pd
import os
from glob import glob
import torch
from torch import nn
import torchvision
import argparse
import os
from torch.nn import functional as F
import albumentations as A
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import parser
from torchvision import transforms, utils
import cv2
from torch.nn import Sequential
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from  tqdm import tqdm
import random
import copy
from datetime import datetime
import segmentation_models_pytorch as smp

import numpy as np
import torchvision.models as models
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1,2,3,4,5,6,7,8'

def get_path_list(tvt_df,key,base_path):
    name_list = tvt_df.loc[tvt_df['tvt']==key,'name'].tolist()
    path_list = []
    for name_ in name_list:
        path_list += glob(os.path.join(base_path,'patch',name_+'*.png'))
    return path_list

def get_augmentation():
    _transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    return A.Compose(_transform)

def get_preprocessing():
    _transform = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    return A.Compose(_transform)

class dataset(nn.Module):
    
    def __init__(self, d, augmentation=None, preprocessing=None):
        self.image_path_li = d
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __getitem__(self, idx):
        img = cv2.imread(self.image_path_li[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.image_path_li[idx].replace('patch','mask'),0)
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
            
        return img, mask.unsqueeze(0)
    
    def __len__(self):
        return len(self.image_path_li) 

def main(args):
    BATCH_SIZE  = [args.batch_size, 1]
    tvt_df = pd.read_csv(args.split_csv)
    train_path_list = get_path_list(tvt_df,'train',args.base_path)
    valid_path_list = get_path_list(tvt_df,'valid',args.base_path)
    train_dataset = dataset(train_path_list,augmentation = get_augmentation(),preprocessing = get_preprocessing())
    valid_dataset = dataset(valid_path_list,preprocessing = get_preprocessing())
    # numworker 설정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE[1], shuffle=False, num_workers=1)
    
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
    ]
    
    seg_model = smp.Unet(
    encoder_name = 'resnet50',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet", # use `imagenet` pretrained weights for encoder initialization
    activation = 'sigmoid',
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    )
    seg_model.load_state_dict(torch.load(args.save_model,map_location='cuda')['model_state'])
    seg_model.to(args.device)
    
    optimizer = torch.optim.Adam([ 
        dict(params=seg_model.parameters(), lr=args.lr),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
    seg_model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=args.device,
    verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        seg_model, 
        loss=loss, 
        metrics=metrics, 
        device=args.device,
        verbose=True,
    )
    
    
    history_train_loss = []
    history_valid_loss = []
    history_train_iou = []
    history_valid_iou = []
    for i in range(1000):
        train_dataset.random()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True, num_workers=0)
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        history_train_loss.append(train_logs['dice_loss'])
        history_valid_loss.append(valid_logs['dice_loss'])
        history_train_iou.append(train_logs['iou_score'])
        history_valid_iou.append(valid_logs['iou_score'])
        if best_loss > valid_logs['dice_loss']:
            model_state = copy.deepcopy(seg_model.state_dict())
            best_loss = valid_logs['dice_loss']
            ls_cnt=0
        else:
            ls_cnt+=1
            if ls_cnt ==10:
                print('early stop')
                break 
    torch.save({'model_state' : model_state,
                'history_dict': {'train_loss' : history_train_loss,'train_iou' : history_train_iou,
                                'valid_loss' : history_valid_loss,'valid_iou' : history_valid_iou}},
                f'{args.save_model}')

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('-p', '--base-path', type=str)
    parser.add_argumnet('-m', '–save-model' ,type=str)
    parser.add_argument('-c', '--split-csv', type=str)
    parser.add_argument('-d', '--device', default=0, type=int)
    parser.add_argument('-b', '--batch-size',default=8, type=int)
    parser.add_argument('-l', '--lr',default = 0.0001, type=float)

    args = parser.parse_args()
    main(args)