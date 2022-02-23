import os
from utils import time_reset
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from U_Net_model import UNET
from utils import get_loaders, save_predictions_as_imgs, load_checkpoint
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import numpy as np
import cv2

from configs import CLOUD_PERCENT_PATH, CHECKPOINT_PATH, SUBREGION_PATH, \
    PREDICT_DATA_PATH, PREDICTED_IMG_PATH, PREDICTED_PREDS_PATH

from configs import IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, \
    NUM_WORKERS, PIN_MEMORY, LOAD_MODEL, TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_REGION_DIR, \
    VAL_IMG_DIR, VAL_MASK_DIR, VAL_REGION_DIR, TEST_IMG_DIR, TEST_MASK_DIR, TEST_REGION_DIR

if not os.path.exists(PREDICTED_IMG_PATH):
    os.mkdir(PREDICTED_IMG_PATH)
if not os.path.exists(PREDICTED_PREDS_PATH):
    os.mkdir(PREDICTED_PREDS_PATH)

#clear old records
for harbor in os.listdir(PREDICTED_IMG_PATH):
    for old_name in os.listdir(os.path.join(PREDICTED_IMG_PATH, harbor)):
        os.remove(os.path.join(PREDICTED_IMG_PATH, harbor, old_name))
for harbor in os.listdir(PREDICTED_PREDS_PATH):
    for old_name in os.listdir(os.path.join(PREDICTED_PREDS_PATH, harbor)):
        os.remove(os.path.join(PREDICTED_PREDS_PATH, harbor, old_name))

features = [64, 128, 256, 512]
model = UNET(features=features, in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load(CHECKPOINT_PATH), model)
model.eval()

df = pd.read_csv(CLOUD_PERCENT_PATH)
search_dict = {}
for i in range(df.shape[0]):
    line = df.loc[i, :]
    search_dict[line['harbor']+time_reset(line['name'])] = line['cloud_all_percent']

count = 0

for harbor in os.listdir(PREDICT_DATA_PATH):
    mask = cv2.imread(os.path.join(SUBREGION_PATH, harbor+'.png'))
    mask = cv2.split(cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH)))[2]
    mask[mask != 0] = 1

    if not os.path.exists(os.path.join(PREDICTED_PREDS_PATH, harbor)):
        os.mkdir(os.path.join(PREDICTED_PREDS_PATH, harbor))
    if not os.path.exists(os.path.join(PREDICTED_IMG_PATH, harbor)):
        os.mkdir(os.path.join(PREDICTED_IMG_PATH, harbor))

    for name in os.listdir(os.path.join(PREDICT_DATA_PATH, harbor)):
        if search_dict[harbor + name] < 0.05:
            count += 1
            print('predicting', harbor, count)
            transform = transforms.Compose(
                [
                 transforms.ToTensor(), ]
            )
            img = cv2.imread(os.path.join(PREDICT_DATA_PATH, harbor, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (480, 480))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = transform(img)
            img = img.unsqueeze(0)
            img = img.to(DEVICE)

            with torch.no_grad():
                preds = torch.sigmoid(model(img))
                preds = (preds > 0.5).float().cpu()

            preds = preds * mask

            cv2.imwrite(os.path.join(PREDICTED_IMG_PATH, harbor + '_' + name), np.array(preds[0].repeat(3, 1, 1).permute(1, 2, 0)) * 255)

            container_sum = np.sum(np.array(preds))

            line = [harbor, name, container_sum]

            with open('predict_results.txt', 'a+') as f:
                f.write(harbor + ',' + name + ',' + str(container_sum))
                f.write('\n')