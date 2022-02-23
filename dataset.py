import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import torchvision
from configs import IMAGE_HEIGHT, IMAGE_WIDTH

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, label_dir, region_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.region_dir = region_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index].replace(".jpg", "_mask.gif"))
        region_path = os.path.join(self.region_dir, self.images[index])

        image = cv2.resize(np.array(Image.open(img_path).convert("RGB")), (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(np.array(Image.open(label_path).convert("L"), dtype=np.float32), (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        label[label == 255.0] = 1.0

        region = np.array(cv2.split(cv2.resize(cv2.imread(region_path), (IMAGE_WIDTH, IMAGE_HEIGHT),  interpolation=cv2.INTER_NEAREST))[0], dtype=np.float32)
        region[region != 0.0] = 1.0

        stack = np.zeros([IMAGE_WIDTH, IMAGE_HEIGHT, 5])
        stack[:, :, :3] = image
        stack[:, :, 3] = label
        stack[:, :, 4] = region

        if self.transform is not None:
            augmentations = self.transform(image=stack)
            stack = augmentations["image"]
            image = stack[:3, :, :].type(torch.float)
            label = stack[3, :, :].type(torch.float)
            region = stack[4, :, :].type(torch.float)

        return image, label, region


