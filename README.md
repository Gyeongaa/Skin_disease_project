# Skin_disease_project
Development of lesion segmentation model using skin disease pathological examination image data. (2022)


### Description
Pretrained model setting
- Model: U-Net
- Model encoder block: ResNet50
- Patch size: 512 x 512 pixel (x50)
- Augmentation: horizontal flip, vertical flip
- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: Dice loss
- Batch size: 16
- Average dice score: 0.90

#### PatchExtraction.py
Extract patch and mask images used for segmentation model learning based on lesion area (Layer2) in WSI and annotation file (.json)   
**Optional argument**    
-i --init-dir: Path where the source data and labeling data directories exist   
-d --save-dir: Path to save patch and mask images.   
-t --task: Task number (1: Epidermoid cyst 2: Seborrheic keratosis 3: Bowen disease/squamous cell carcinoma 4: Basal cell carcinoma 5: Melanocytic nevus 6: Melanoma)    
-l --level: scale (0: 200x 1: 100x 2: 50x 3: 25x 4: 12.5x), default =2   
-p --patch-size: Patch image size (pixel), default = 512    
-s --stride: Interval for extracting patches within the slide image (ratio to image size), default = 1.0   

#### Train.py
Train segmentation model    
**Optional argument**    
-p --base-path: Path where patch and mask directories exist   
-c --split-csv: Train dataset, valid dataset, test dataset split table (.csv)   
-m --save-model: Learning model storage path    
-d --device: GPU number, default=0   
-b --batch-size: Batch size, default=8    
-l --lr: Learning rate, default=0.0001    


#### Learning model (.pth) configuration 
You can check the learning model results by accessing the learning model parameters (key: model_state) and learning curve (key: history_dict) in dictionary form.   
Inference.py: Check test set performance   
**Optional argument   **
-i –init-dir: Path where the source data and labeling data directories exist    
-t –task: task number default= task1    
-c –split-csv: Train dataset, valid dataset, test dataset split table (.csv)    
-m –path-model: Evaluation model file path    
-d –device: GPU number, default=0    
-r –result_path: Test set result storage path (.txt)    

 
