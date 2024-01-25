import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import tifffile as tiff
import cv2


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, img_size = 640, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(self, mask_file[0])
        img = load_image(self, img_file[0])

        # Remove dim & convert to 0-1 value mask
        if len(mask.shape) == 3:
            mask = np.clip(mask[0, :, :], 0, 1)
        
        try:
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
        except Exception as e:
            print('ERROR: ', name)
            print(e)

        return {
            'image': img,
            'mask': mask,
            'name' : name
        }

def load_image(self, filename):
    ext = splitext(filename)[1]
    if ext in ['.tiff']:
        img = tiff.imread(filename) # Band: R,G,B,NIR
        img = np.array(img, dtype=np.float32)
        if img.shape[2] <= 4:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        elif img.shape[2] == 6:
            temp1 = img[:,:,:3]
            temp1 = cv2.resize(temp1, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            temp2 = img[:,:,3:]
            temp2 = cv2.resize(temp2, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            img = np.concatenate((temp1, temp2), axis=2)
        else: 
            raise ValueError(f'Input image should have 3, 4 or 6 bands, found {img.shape[2]}')
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        return img
    elif ext in ['.png', '.jpg', '.jpeg']:
        # Load mask
        mask = cv2.imread(str(filename))  # BGR
        assert mask is not None, "Image Not Found"
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = mask[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        mask = np.ascontiguousarray(mask)
        return mask
    else:
        raise ValueError(f'Input image should be .tiff, found {ext}')
