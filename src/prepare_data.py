"""
resizing and cropping images to 1024x680.
"""
from __future__ import print_function, division

import os
from os.path import join
import random
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pypfm import PFMLoader
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import ImageOps

import utils as defs
import visualize as viz
from utils import get_depth_dir, get_img_dir, get_test_dir
from PIL import Image
import os
from os.path import join
from matplotlib import pyplot as plt
from visualize import blend_images
from data_loader import GeoposeDataset, GeoposeToTensor


# Function to change the image size
def change_image_size(maxWidth,
                      maxHeight,
                      image):
    widthRatio = maxWidth / image.size[0]
    heightRatio = maxHeight / image.size[1]

    newWidth = int(widthRatio * image.size[0])
    newHeight = int(heightRatio * image.si  ze[1])

    newImage = image.resize((newWidth, newHeight))
    return newImage


class ResizeDepth:
    def __call__(self, sample):
        """
        resize depth to be same resolution as image it's giving depth info on.
        Returns: sample

        Args:
            sample: data sample with 'depth' and 'image'.
        """
        image, depth, seg = sample['image'], sample['depth'], sample['segmap']
        # plt.imshow(depth.squeeze().numpy())
        # plt.show()
        depth_resized = TF.resize(depth, (image.shape[1], image.shape[2]))
        seg_resized = TF.resize(seg, (image.shape[1], image.shape[2]))
        # TODO: bilinear interpolation? good or bad in my case?
        #   for sure screws up the -1's at the end.. Fix later.
        # print(f'old max: {depth.max()} old min: {depth.min()}\n'
        #       f'new max: {depth_resized.max()}, new min: {depth_resized.min()}')
        # plt.imshow(np_resized)
        # plt.show()
        # blended = blend_images(TF.to_pil_image(image), TF.to_pil_image(depth_resized))
        # plt.imshow(blended)
        # plt.show()
        sample['depth'] = depth_resized
        sample['segmap'] = seg_resized
        return sample


class CropToAspectRatio:
    """
    crop sides so as to get the required aspect ratio.
    the image then can just be rescaled to required resolution.
    """

    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def __call__(self, imgs):
        for img in imgs:
            # if img.size
            pass
            # TODO: cropsides
            # img = TF.resize(img, (self.xres, self.yres)) - just use transform.Resize.
        return imgs


# TODO: afterward:
#  if img>resolution: TF.center_crop()
#  elif img<resolution: TF.pad()
#  also: just use PIL.ImageOps.fit for resizing instead of shady thing I found online.
#  and for cropping to AR just do some simple math.
#  also don't forget mask for sky!


class CenterAndCrop:
    def __init__(self, h, w):
        """
        transform images to all be same resolution hxw.

        Args:
            h: height (x?)
            w: width (y?)
        """
        self.h = h
        self.w = w

    def __call__(self, sample):
        for k, v in sample.items():
            if torch.is_tensor(v):
                # plt.imshow(v.numpy().transpose((1, 2, 0)))
                # plt.show()
                sample[k] = TF.center_crop(v, (self.h, self.w))
                # plt.imshow(sample[k].numpy().transpose((1, 2, 0)))
                # plt.show()
        return sample


if __name__ == '__main__':
    geoset = GeoposeDataset(transform=transforms.Compose([GeoposeToTensor(),
                                                          ResizeDepth(),
                                                          CenterAndCrop(680, 1024)
                                                          ]))
    geoset_no_crop = GeoposeDataset(transform=transforms.Compose([GeoposeToTensor(),
                                                                  ResizeDepth()]))
    dl = DataLoader(geoset, batch_size=4)
    batch = next(iter(dl))
    s2 = next(iter(geoset_no_crop))
    # print single sample,
    # since if we don't crop we can't have them all in one batch.
    for k, v in s2.items():
        if torch.is_tensor(v):
            s2[k] = v.unsqueeze(0)
        elif type(v).__name__ == 'str_':
            s2[k] = [v]
    viz.show_batch(batch)
    plt.show()
    viz.show_batch(s2)
    plt.show()
    # rd = ResizeDepth()
    # rd(sample)
