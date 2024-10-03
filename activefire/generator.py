import threading
import random
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as shuffle_lists
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob 
images_path='../../../../dataset/images/patches/'
masks_path='../../../../dataset/masks/intersection/' 
MAX_PIXEL_VALUE = 65535  # Max. pixel value, used to normalize the image

class ThreadSafeIter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))
    return g

def get_img_arr(path):
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))  # HWC format
        img = np.float32(img) / MAX_PIXEL_VALUE
    return img

def get_mask_arr(path):
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))  # HWC format
        seg = np.float32(img)
    return seg

class CustomDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image = get_img_arr(self.images_path[idx])
        mask = get_mask_arr(self.masks_path[idx])

        if self.transform and callable(self.transform):
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            raise TypeError('The transform function should be callable and return a dictionary containing image and mask.')

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # HWC -> CHW
        return image, mask


transform = A.Compose([
    A.Resize(height=128, width=128) # Convert numpy arrays to tensors
])
def create_dataloader(images_path, masks_path, batch_size=32, shuffle=True, random_state=None, num_workers=4, transform=None):
    dataset = CustomDataset(images_path, masks_path, transform=transform)
    
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        images_path, masks_path = shuffle_lists(images_path, masks_path)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

train_loader = create_dataloader(images_path, masks_path, batch_size=16, shuffle=True, random_state=42, transform=transform)
