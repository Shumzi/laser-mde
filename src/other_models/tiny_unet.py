import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cfg

class UNetConvBlock(nn.Module):
    """
    this block computes a double conv layer, as described in the blue arrows in unet.
    layer channel sizes are: input (in_size) -> middle (out_size) -> out (out_size).
    max pooling (down sampling) is done outside of the block.
    """
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        if cfg['model']['arch']['use_bn']:
            self.bn = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        if cfg['model']['arch']['use_bn']:
            out = self.bn(out)
        return out


class UNetUpBlock(nn.Module):
    """
    this block does upsampling for the second half of the unet.
    stages in block:
    1. input is upsampled with a transpose conv
    2. upsampled layer is now concatted with the corresponding layer from the down stage.
    3. double conv on concatted layer (same as convblock).
    """
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation
        if cfg['model']['arch']['use_bn']:
            self.bn = nn.BatchNorm2d(out_size)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.activation(self.conv(out))
        out = self.conv2(out)
        if cfg['model']['arch']['use_bn']:
            out = self.bn(out)
        return out


class UNet(nn.Module):
    """
    Mini UNet arch. This arch downsamples to 1/8 of the image size (as opposed to 1/16 in original UNet),
    and uses less channels (4x less per layer - e.g. 16 in layer 1 instead of 64 in original paper).
    Used as a toy net to see that it works at all.
    """
    def __init__(self):
        super().__init__()

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)

        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        block1 = self.conv_block3_16(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block16_32(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block32_64(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block64_128(pool3)

        up1 = self.activation(self.up_block128_64(block4, block3))

        up2 = self.activation(self.up_block64_32(up1, block2))

        up3 = self.up_block32_16(up2, block1)

        last = self.last(up3)
        
        return torch.sigmoid(last.squeeze())