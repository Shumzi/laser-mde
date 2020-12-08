import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb


# making unet with 1channel output. eigen_net uses way more parameters,
# don't think my gpu can handle it.


class UNetConvBlock(nn.Module):
    """
    this block does the double conv in the unet paper.
    downsampling and upsampling are done outside this module.
    """

    def __init__(self, in_channel, out_channel):
        super(UNetConvBlock, self).__init__()

    # def forward(self, x):


class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)