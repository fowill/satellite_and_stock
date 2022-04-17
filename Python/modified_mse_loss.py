import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target, region):
        input = torch.sigmoid(input)
        loss = ((input - target)**2 * region).sum() / region.sum()
        return loss