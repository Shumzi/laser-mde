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
        if cfg['model']['use_double_bn']:
            self.bn1 = nn.BatchNorm2d(out_size)
        if cfg['model']['use_bn']:
            self.bn2 = nn.BatchNorm2d(out_size)
        if cfg['model']['dropout'] is not None:
            self.dropout = nn.Dropout2d(p=cfg['model']['dropout'])

    def forward(self, x):
        out = self.activation(self.conv(x))
        if cfg['model']['use_double_bn']:
            out = self.bn1(out)
        out = self.activation(self.conv2(out))
        if cfg['model']['use_bn']:
            out = self.bn2(out)
        if cfg['model']['dropout'] is not None:
            out = self.dropout(out)
        return out


class UNetUpBlock(nn.Module):
    """
    this block does upsampling for the second half of the unet.
    stages in block:
    1. input is upsampled with a transpose conv
    2. upsampled layer is now concatted with the corresponding layer from the down stage.
    3. double conv on concatted layer (same as convblock).
    regularization stages are as described in https://arxiv.org/pdf/1904.03392.pdf
    """

    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv_block = UNetConvBlock(in_size, out_size, kernel_size, activation)

    def forward(self, x, bridge):
        up = self.up(x)
        # if bridge.shape != up.shape:
        #     bridge = bridge[:, :, :up.shape[2], :]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):
    """
    Mini UNet arch. This arch downsamples to 1/8 of the image size (as opposed to 1/16 in original UNet),
    and uses less channels (4x less per layer - e.g. 16 in layer 1 instead of 64 in original paper).
    Used as a toy net to see that it works at all.
    """

    def __init__(self, in_channel=3, tiny=True, sigmoid=True):
        super().__init__()
        if tiny:
            mult = 1
        else:
            mult = 4
        self.sigmoid = sigmoid
        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(in_channel, 16 * mult)
        self.conv_block16_32 = UNetConvBlock(16 * mult, 32 * mult)
        self.conv_block32_64 = UNetConvBlock(32 * mult, 64 * mult)
        self.conv_block64_128 = UNetConvBlock(64 * mult, 128 * mult)

        self.up_block128_64 = UNetUpBlock(128 * mult, 64 * mult)
        self.up_block64_32 = UNetUpBlock(64 * mult, 32 * mult)
        self.up_block32_16 = UNetUpBlock(32 * mult, 16 * mult)

        self.last = nn.Conv2d(16 * mult, 1, 1)

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
        if self.sigmoid:
            return torch.sigmoid(last)
        else:
            return last
