import argparse
import json
import os
from glob import glob
from tqdm import tqdm
import openslide
import numpy as np
import cv2
import argparse
import warnings

LEVEL_RESOLUTION = {
    0 : 1,
    1 : 2, # 1/2 resoltion
    2 : 4, # 1/4 resolution
    3 : 8, # 1/4 resolution
    4 : 16, # 1/16 resolution
}

def main(args):
    if os.path.isdir(args.save_dir):
        os.makedirs(os.path.join(args.save_dir,'patch'),exist_ok=True)
        os.makedirs(os.path.join(args.save_dir,'mask'),exist_ok=True)
    
    #json base path
    json_path_list = glob(os.path.join(args.init_dir,f"라벨링데이터/0{args.task}/*/*/*.json"))

    for path_ in tqdm(sorted(json_path_list)):
        name_ = path_.split('/')[-1].replace('.json','')
        #slide base path
        base_slide_path = path_.replace('라벨링데이터','원천데이터').replace('.json','')
        try:
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
                raise Exception('{}: 해당 경로의 파일 없음.'.format(base_slide_path))
        except:
            print('{}: 읽기 실패'.format(base_slide_path))
            continue

        level_list = [round(i) for i in wsi_slide.level_downsamples]
        level_idx = level_list.index(LEVEL_RESOLUTION[args.level])
    #     print(level_idx)
        #check mpp
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
                x_,y_ = cord_['X']//int(LEVEL_RESOLUTION[args.level]*MPP_W), cord_['Y']//int(LEVEL_RESOLUTION[args.level]*MPP_W)
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
                x_,y_ = cord_['X']//int(LEVEL_RESOLUTION[args.level]*MPP_W), cord_['Y']//int(LEVEL_RESOLUTION[args.level]*MPP_W)
                bound_.append([x_,y_])
            cnt_ = np.array(bound_).reshape((-1, 1, 2)).astype(np.int32)
            if region_['NegativeROA']== 0:
                    cv2.fillPoly(layer2_mat_posi, [cnt_], 1)
            elif region_['NegativeROA']== 1:
                cv2.fillPoly(layer2_mat_nega, [cnt_], 1)
        layer2_mat_posi[layer2_mat_nega==1] = 0

        for y0 in range(0,wsi_h,int(args.patch_size*args.stride)):
            for x0 in range(0,wsi_w,int(args.patch_size*args.stride)):
                if not ((y0+args.patch_size) > wsi_h) or ((x0+args.patch_size) > wsi_w):
                    core_ = np.uint8(layer1_mat_posi[y0:y0+args.patch_size,x0:x0+args.patch_size])
                    mask_ = np.uint8(layer2_mat_posi[y0:y0+args.patch_size,x0:x0+args.patch_size])
                    patch_ = np.uint8(wsi_array[y0:y0+args.patch_size,x0:x0+args.patch_size])
                    if np.sum(core_)/(args.patch_size**2) > 0.3:
                                    cv2.imwrite(os.path.join(args.save_dir,'patch',f'{name_}_X0_{int(x0*LEVEL_RESOLUTION[args.level]*MPP_W)}_Y0_{int(y0*LEVEL_RESOLUTION[args.level]*MPP_W)}.png'),cv2.cvtColor(patch_,cv2.COLOR_RGB2BGR))
                                    cv2.imwrite(os.path.join(args.save_dir,'mask',f'{name_}_X0_{int(x0*LEVEL_RESOLUTION[args.level]*MPP_W)}_Y0_{int(y0*LEVEL_RESOLUTION[args.level]*MPP_W)}.png'),mask_)
                else:
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task1 patch extraction')
    parser.add_argument('-i', '--init-dir', type=str)
    parser.add_argument('-d', '--save-dir', type=str)
    parser.add_argument('-t', '--task',default=1,type=int)
    parser.add_argument('-l', '--level',default=2, type=int)
    parser.add_argument('-p', '--patch_size',default=512, type=int)
    parser.add_argument('-s', '--stride',default=2.0, type=float)
    args = parser.parse_args()
    main(args)
