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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
