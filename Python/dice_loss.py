import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, mask):

        input = torch.sigmoid(input)

        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        mask_flat = mask.view(N, -1)

        intersection = input_flat * target_flat * mask_flat

        loss = 2 * (intersection.sum() + smooth) / ((input_flat*mask_flat).sum() + (target_flat*mask_flat).sum() + smooth)
        loss = 1 - loss.sum() / N

        return loss