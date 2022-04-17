import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target, mask):
        input = torch.sigmoid(input)
        loss = -1*(target*torch.log(input)+(1-target)*torch.log(1-input))*mask
        loss = loss.mean()

        return loss