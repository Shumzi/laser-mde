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