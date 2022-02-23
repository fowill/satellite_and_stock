import torch
import os
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from U_Net_model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

from modified_mse_loss import MSELoss
import cv2
from torch.utils.data import random_split, SubsetRandomSampler, ConcatDataset
from torch.utils.data import DataLoader
from dataset import CarvanaDataset

from sklearn.model_selection import KFold
import numpy as np

from configs import IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, \
    NUM_WORKERS, PIN_MEMORY, LOAD_MODEL, TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_REGION_DIR, \
    VAL_IMG_DIR, VAL_MASK_DIR, VAL_REGION_DIR, TEST_IMG_DIR, TEST_MASK_DIR, TEST_REGION_DIR,\
    TRAINING_MODEL_SAVE_DIR,\
    K_FOLD_NUM


def train_fn(epoch, fold, loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    count = 0
    for batch_id, (data, target, region) in enumerate(loop):

        data = data.to(device=DEVICE)
        target = target.float().unsqueeze(1).to(device=DEVICE)
        region = region.float().unsqueeze(1).to(device=DEVICE)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, target, region)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        count += 1

    count = 0


train_transforms = A.Compose(
    [
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # ToTensor doesn't divide by 255 lik PyTorch,
        # it's done inside Normalize function
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ],
)

train_ds = CarvanaDataset(
    image_dir=TRAIN_IMG_DIR,
    label_dir=TRAIN_MASK_DIR,
    region_dir=TRAIN_REGION_DIR,
    transform=train_transforms,
)
val_ds = CarvanaDataset(
    image_dir=VAL_IMG_DIR,
    label_dir=VAL_MASK_DIR,
    region_dir=VAL_REGION_DIR,
    transform=train_transforms,
)
test_ds = CarvanaDataset(
    image_dir=TEST_IMG_DIR,
    label_dir=TEST_MASK_DIR,
    region_dir=TEST_REGION_DIR,
    transform=val_transforms,
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)

dataset = ConcatDataset([train_ds, val_ds])

splits = KFold(n_splits=K_FOLD_NUM, shuffle=True, random_state=23)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ff = [[64, 128], [64, 128, 256], [64, 128, 256, 512], [64, 128, 256, 512, 1024]]
fnames = ['128', '256', '512', '1024']

for i in range(len(ff)):
    f = ff[i]
    fname = fnames[i]

    for epoch in range(NUM_EPOCHS):

        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

            model = UNET(features=f, in_channels=3, out_channels=1).to(device)
            loss_fn = MSELoss()
            scaler = torch.cuda.amp.GradScaler()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            print('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True,)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True,)

            train_fn(epoch, fold, train_loader, model, optimizer, loss_fn, scaler, )

            #save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            shutil.copy(r'./my_checkpoint.pth.tar', TRAINING_MODEL_SAVE_DIR)
            os.rename(os.path.join(TRAINING_MODEL_SAVE_DIR, 'my_checkpoint.pth.tar'), os.path.join(TRAINING_MODEL_SAVE_DIR, str(epoch) + '-' + str(fold) + 'my_checkpoint.pth.tar'))

            # check accuracy on val set
            check_accuracy('val', train_loader, model, fname, device=DEVICE)
            
            # check accuracy on test set
            check_accuracy('test', test_loader, model, fname, device=DEVICE)
