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
