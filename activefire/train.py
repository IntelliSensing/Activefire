import os
import warnings
warnings.filterwarnings("ignore")

import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

from generator import CustomDataset  
from models import UNet  
from plot_history import plot_history

import pandas as pd
import random
import time

# if True plot the training and validation graphs
PLOT_HISTORY = True 

MASK_ALGORITHM = 'Intersection'

N_FILTERS = 64
N_CHANNELS = 10

EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = (256,256)
MODEL_NAME = 'unet'

RANDOM_STATE = 42

IMAGES_PATH = '../../../../dataset/images/patches/'
MASKS_PATH = '../../../../dataset/masks/intersection/'  
OUTPUT_DIR = './train_output/' 

WORKERS = 4

EARLY_STOP_PATIENCE = 5 

CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}.pt'.format(MODEL_NAME, MASK_ALGORITHM)

# If not zero will be load as weights
INITIAL_EPOCH = 0
RESTART_FROM_CHECKPOINT = None
if INITIAL_EPOCH > 0:
    RESTART_FROM_CHECKPOINT = os.path.join(OUTPUT_DIR, 'checkpoint-{}-{}-epoch_{:02d}.pt'.format(MODEL_NAME, MASK_ALGORITHM, INITIAL_EPOCH))

FINAL_WEIGHTS_OUTPUT = './final_weights.pth/'

CUDA_DEVICE = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(CUDA_DEVICE)

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass

x_train = pd.read_csv('./dataset/images_train.csv')
y_train = pd.read_csv('./dataset/masks_train.csv')
x_val = pd.read_csv('./dataset/images_val.csv')
y_val = pd.read_csv('./dataset/masks_val.csv')
x_test = pd.read_csv('./dataset/images_test.csv')
y_test = pd.read_csv('./dataset/masks_test.csv')

# Map the images and mask path
images_train = [os.path.join(IMAGES_PATH, image) for image in x_train['images']]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in y_train['masks']]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['images']]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in y_val['masks']]

# Define the transform function
transform = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]) # Convert numpy arrays to tensors
])

# Define dataset and dataloader
train_dataset = CustomDataset(images_train, masks_train, transform=transform)
val_dataset = CustomDataset(images_validation, masks_validation, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

# Define model
model = UNet(in_channels=N_CHANNELS, n_classes=1, n_filters=N_FILTERS)
model = nn.DataParallel(model)  # Use data parallelism
model.to(device)

# Define loss function (Dice Loss + CE Loss)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1 - dice

dice_loss = DiceLoss()
ce_loss = nn.BCEWithLogitsLoss()

def combined_loss(y_pred, y_true):
    return dice_loss(y_pred, y_true) + ce_loss(y_pred, y_true)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=EARLY_STOP_PATIENCE, verbose=True)

# Load existing model weights
if INITIAL_EPOCH > 0 and RESTART_FROM_CHECKPOINT:
    model.load_state_dict(torch.load(RESTART_FROM_CHECKPOINT))

# Training loop
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(INITIAL_EPOCH, EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save model weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME.replace('.pt', f'-epoch_{epoch + 1:02d}.pt')))


print('Train finished!')

# Save final model weights
torch.save(model.state_dict(), FINAL_WEIGHTS_OUTPUT)
print("Weights Saved: {}".format(FINAL_WEIGHTS_OUTPUT))

if PLOT_HISTORY:
    plot_history(train_losses, val_losses, OUTPUT_DIR)
