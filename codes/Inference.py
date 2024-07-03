import json
import os
from glob import glob
from tqdm import tqdm
import openslide
import numpy as np
import cv2
import argparse
import segmentation_models_pytorch as smp
from datetime import datetime
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import warnings
import pandas as pd

# GPU 지정
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1,2,3,4,5,6,7,8'
warnings.filterwarnings("ignore")

LEVEL_IDX ={
    'level0' : 0,
    'level2' : 1,
    'level4' : 2,
}
LEVEL_RESOLUTION = {
    'level0' : 1,
    'level1' : 2, # 1/2 resoltion
    'level2' : 4, # 1/4 resolution
    'level3' : 8, # 1/4 resolution
    'level4' : 16, # 1/16 resolution
}
PATCH_SIZE=  512
EXTRACTION_LEVEL = 'level2'

def search_file(base_path, format_):
    list_ = []
    for (path, dir, files) in os.walk(base_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == format_:
                list_.append(os.path.join(path,filename))
    return list_

# https://github.com/qubvel/segmentation_models#documentation 참조
def load_model(model_path,device=0):
    seg_model = smp.Unet(
    encoder_name = 'resnet50',        
    encoder_weights="imagenet", 
    activation = 'sigmoid',
    in_channels=3,                
    classes=1,                    
    )
    seg_model.load_state_dict(torch.load(model_path,map_location=torch.device(device))['model_state'])
    seg_model.to(device)
    return seg_model

def get_preprocessing():
    _transform = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    return A.Compose(_transform)

# Tissue extraction
def get_tissue(img_):
    tissue_kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    org_shape = img_.shape
    target_ = cv2.resize(img_,(img_.shape[1]//8,img_.shape[0]//8))
    _,_,v = cv2.split(cv2.cvtColor(target_, cv2.COLOR_RGB2YUV)) # YUV: Y (영상의 밝기), UV (색체정보) / cvtColor: 색상 공간 변환 / split: 채널 분리
    v = cv2.medianBlur(v, 15)
    _, timg_th = cv2.threshold(v, 138, 255, cv2.THRESH_BINARY)
    timg_th = cv2.dilate(timg_th, tissue_kernel_1, iterations=3)
    tissue_ = cv2.resize(timg_th,(org_shape[1],org_shape[0]))
    return tissue_

def main(args):
    df = pd.read_csv(args.path_csv)
    li = list(df.loc[df['tvt']=='test']['name'])
    json_path_list = []
    for i in li:
        level1, level2,level3 = i.split('_')
        # 원천데이터, 라벨링데이터 폴더가 있는 경로
        json_path_list.append(f'{args.init_dir}/{args.task}/라벨링데이터/{level1}/{level1}_{level2}/{level1}_{level2}_{level3}/{i}.json')
    result_list = []
    loss = smp.utils.losses.DiceLoss()
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for path_ in tqdm(sorted(json_path_list)):
        name_ = path_.split('/')[-1].replace('.json','')
        base_slide_path = path_.replace('라벨링데이터','원천데이터').replace('.json','')
        if os.path.isfile(base_slide_path+'.svs') == True:
            wsi_slide = openslide.OpenSlide(base_slide_path+'.svs')
            mpp = round(float(wsi_slide.properties['aperio.MPP']),2)
        elif os.path.isfile(base_slide_path+'.mrxs') == True:
            wsi_slide = openslide.OpenSlide(base_slide_path+'.mrxs')
            mpp = round(float(wsi_slide.properties['openslide.mpp-x']),2)
        elif os.path.isfile(base_slide_path+'.tif') == True:
            wsi_slide = openslide.OpenSlide(base_slide_path+'.tif')
            mpp = round(float(wsi_slide.properties['openslide.comment'].split('MPP=')[-1]),2)
        else:
            raise Exception('{} 해당 경로의 파일 없음.'.format(base_slide_path))
        
        level_list = [round(i) for i in wsi_slide.level_downsamples]
        level_idx = level_list.index(LEVEL_RESOLUTION[EXTRACTION_LEVEL])
        
        model = load_model(args.path_model,args.device)
        model.eval()
        if mpp > 0.40:
            wsi_w,wsi_h = wsi_slide.level_dimensions[level_idx]
            wsi_array = np.array(wsi_slide.read_region([0,0], level_idx, [wsi_w, wsi_h]))[:,:,:3]
            MPP_W = 1
        else:
            wsi_w,wsi_h = wsi_slide.level_dimensions[level_idx]
            wsi_array = np.array(wsi_slide.read_region([0,0], level_idx, [wsi_w, wsi_h]))[:,:,:3]
            wsi_array = cv2.resize(wsi_array, dsize=(0,0),fx =0.5,fy = 0.5, interpolation=cv2.INTER_AREA)
            wsi_h,wsi_w = wsi_array.shape[:-1]
            MPP_W = 2

            #json setting
        with open (path_, "r", encoding="utf-8-sig") as jsonfile:
            dict_ = json.load(jsonfile)
            layer_1= dict_['Layer1']
            layer_2= dict_['Layer2']
        #layer1
        layer1_mat_posi = np.zeros((wsi_h,wsi_w))
        layer1_mat_nega = np.zeros((wsi_h,wsi_w))
        
        for region_ in layer_1:
            bound_ = []
            for cord_ in region_['Region']:
                x_,y_ = cord_['X']//int(LEVEL_RESOLUTION[EXTRACTION_LEVEL]*MPP_W), cord_['Y']//int(LEVEL_RESOLUTION[EXTRACTION_LEVEL]*MPP_W)
                bound_.append([x_,y_])
            cnt_ = np.array(bound_).reshape((-1, 1, 2)).astype(np.int32)
            if region_['NegativeROA']== 0:
                    cv2.fillPoly(layer1_mat_posi, [cnt_], 1)
            elif region_['NegativeROA']== 1:
                cv2.fillPoly(layer1_mat_nega, [cnt_], 1)
        layer1_mat_posi[layer1_mat_nega==1] = 0
        #layer2
        layer2_mat_posi = np.zeros((wsi_h,wsi_w))
        layer2_mat_nega = np.zeros((wsi_h,wsi_w))
        for region_ in layer_2:
            bound_ = []
            for cord_ in region_['Region']:
                x_,y_ = cord_['X']//int(LEVEL_RESOLUTION[EXTRACTION_LEVEL]*MPP_W), cord_['Y']//int(LEVEL_RESOLUTION[EXTRACTION_LEVEL]*MPP_W)
                bound_.append([x_,y_])
            cnt_ = np.array(bound_).reshape((-1, 1, 2)).astype(np.int32)
            if region_['NegativeROA']== 0:
                    cv2.fillPoly(layer2_mat_posi, [cnt_], 1)
            elif region_['NegativeROA']== 1:
                cv2.fillPoly(layer2_mat_nega, [cnt_], 1)
        layer2_mat_posi[layer2_mat_nega==1] = 0
        tissue_ = get_tissue(wsi_array)

        pr_mat_ = np.zeros((wsi_h,wsi_w))
        for y0 in range(0,wsi_h,PATCH_SIZE//2):
            for x0 in range(0,wsi_w,PATCH_SIZE//2):
                if not ((y0+PATCH_SIZE) > wsi_h) or ((x0+PATCH_SIZE) > wsi_w):
                    mask_ = np.uint8(layer1_mat_posi[y0:y0+PATCH_SIZE,x0:x0+PATCH_SIZE])
                    patch_ = np.uint8(wsi_array[y0:y0+PATCH_SIZE,x0:x0+PATCH_SIZE])
                    if np.sum(mask_)/(PATCH_SIZE**2):
                        sample = get_preprocessing()(image=patch_)
                        patch_= sample['image'].unsqueeze(0).to(device=args.device)
                        if list(patch_.size()) != [1,3,512,512]:
                            continue
                        with torch.no_grad():
                            pr_= model(patch_)
                            pr_ = (pr_ > 0.5).float().squeeze(0).squeeze(0)
                            pr_ = pr_.detach().cpu().numpy()
                            pr_mat_[y0+(PATCH_SIZE//4):y0+PATCH_SIZE-(PATCH_SIZE//4),x0+(PATCH_SIZE//4):x0+PATCH_SIZE-(PATCH_SIZE//4)]= pr_[(PATCH_SIZE//4):PATCH_SIZE-(PATCH_SIZE//4),(PATCH_SIZE//4):PATCH_SIZE-(PATCH_SIZE//4)]
                else:
                    continue
                    
        pr_mat_[tissue_ ==0] = 0
        pr_mat_[layer1_mat_posi==0] = 0
        
        dice_score = 1-loss(torch.Tensor(layer2_mat_posi).to(args.device),torch.Tensor(pr_mat_).to(args.device))

        confusion_vector = torch.Tensor(pr_mat_)/torch.Tensor(layer2_mat_posi)
        tp = torch.sum(confusion_vector == 1).item()
        fp = torch.sum(confusion_vector == float('inf')).item()
        tn = torch.sum(torch.isnan(confusion_vector)).item()
        fn = torch.sum(confusion_vector == 0).item()
        result_list.append([name_,tp,fp,tn,fn, dice_score.item()])

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(args.path_result,'result.txt'), 'w', encoding='utf-8') as f:
        f.writelines(f'Start time : {start_time}\n')
        avg = 0
        for i in result_list:        
            f.writelines(f'{i[0]} | TP : {i[1]} FP : {i[2]} TN : {i[3]} FN : {i[4]}  Dice score : {round(i[5],4)}\n')
            avg+= i[5]
        f.writelines(f'Average dice score : {round(avg/len(result_list),4)}\n')
        f.writelines(f'End time : {end_time}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference model')
    parser.add_argument('-i', '--init-dir')
    parser.add_argument('-t', '--task', default='task1')
    parser.add_argument('-c', '--path-csv', type=str)
    parser.add_argument('-m', '--path-model', default='/workspace/test/task1_model.pth', type=str)
    parser.add_argument('-d', '--device', default=0, type=int)
    parser.add_argument('-r', '--path-result', default='',type=str)
    args = parser.parse_args()
    main(args)
    