import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os
import cv2
from torch.utils.data import random_split, SubsetRandomSampler
from configs import IMAGE_HEIGHT, IMAGE_WIDTH, TRAINING_RESULT_SAVE_DIR, PREDICTED_IMG_PATH

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    mask_dir,
    mask_maskdir,
    batch_size,
    train_transform,
    val_transform,
    mask_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    mask_ds = CarvanaDataset(
        image_dir=mask_dir,
        mask_dir=mask_maskdir,
        transform=mask_transform
    )
    mask_loader = DataLoader(
        mask_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader, mask_loader

def check_accuracy(mode, loader, model, fname, device="cuda",):

    num_correct = 0
    num_true_correct = 0
    num_true = 0
    num_pixels = 0
    dice_score = 0
    precision = 0
    recall = 0
    f1_score = 0
    model.eval()

    with torch.no_grad():
        count = 0
        for x, y, z in loader:
            region = z
            region = region.to(device)
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += ((preds == y)*region).sum()
            num_true += (y*region).sum()
            num_true_correct += ((preds*y)*region).sum()
            num_pixels += region.sum()
            dice_score += (2*(((preds*y)*region).sum()))/(
                ((preds+y)*region).sum()+1e-8
            )
            count += 1

    precision = num_correct / num_pixels
    recall = num_true_correct / num_true
    f1_score = 2 * precision * recall / (precision + recall)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    print(f'precision:{precision}')
    print(f'recall:{recall}')
    print(f'f1_score:{f1_score}')

    with open(os.path.join(TRAINING_RESULT_SAVE_DIR, fname + '_' + mode + '_performance.txt'), 'a+') as f:
        f.write(str(dice_score/len(loader)))
        f.write(',')
        f.write(str(precision))
        f.write(',')
        f.write(str(recall))
        f.write(',')
        f.write(str(f1_score))
        f.write('\n')

    model.train()

def save_predictions_as_imgs(
        loader, model, folder=PREDICTED_IMG_PATH, device="cuda"
):
    model.eval()
    for idx, (x, y, z) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1),  f"{folder}/{idx}.png")

    model.train()

def time_reset(a):
    check = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12',
    }
    month = check[a.split(' ')[1]]
    day = a.split(' ')[2]
    year = a.split(' ')[3].replace('.png','')
    return year+'-'+month+'-'+day+'.png'

