import os
import rasterio
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import UNet
import pandas as pd
import torch.nn as nn

MASK_ALGORITHM = 'Intersection'
N_FILTERS = 64
N_CHANNELS = 10
IMAGE_SIZE = (256, 256)
MODEL_NAME = 'unet'
TH_FIRE = 0.25
CUDA_DEVICE = 0
IMAGES_PATH = '../../../../dataset/images/patches/'
MASKS_PATH = '../../../../dataset/masks/intersection/'

IMAGES_CSV = './dataset/images_test.csv'
MASKS_CSV = './dataset/masks_test.csv'

OUTPUT_DIR = './log'
WEIGHTS_FILE = './final_weights.pth'
device = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(CUDA_DEVICE)

images_df = pd.read_csv(IMAGES_CSV)
masks_df = pd.read_csv(MASKS_CSV)

images = [os.path.join(IMAGES_PATH, image) for image in images_df['images']]
masks = [os.path.join(MASKS_PATH, mask) for mask in masks_df['masks']]

model = UNet(in_channels=N_CHANNELS, n_classes=1, n_filters=N_FILTERS).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(WEIGHTS_FILE))


transform = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.Normalize(mean=(0.485,) * N_CHANNELS, std=(0.229,) * N_CHANNELS),  
    ToTensorV2() 
])

for image_path, mask_path in zip(images, masks):
    try:
        with rasterio.open(image_path) as src:
            image = src.read()  
            image = np.moveaxis(image, 0, -1) 

        image = image.astype(np.float32)

        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(np.float32) 

        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            y_pred = torch.sigmoid(y_pred).cpu().numpy()[0, 0, :, :] > TH_FIRE

        y_true = mask > TH_FIRE
        txt_mask_path = os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'grd_' + os.path.basename(mask_path).replace('.tif', '.txt'))
        txt_pred_path = os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'det_' + os.path.basename(image_path).replace('.tif', '.txt'))

        np.savetxt(txt_mask_path, y_true.astype(int), fmt='%i')
        np.savetxt(txt_pred_path, y_pred.astype(int), fmt='%i')

    except Exception as e:
        print(e)
        with open(os.path.join(OUTPUT_DIR, "error_log_inference.txt"), "a+") as myfile:
            myfile.write(str(e) + "\n")
