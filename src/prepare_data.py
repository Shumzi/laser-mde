import os
from os.path import join
import random
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# from pypfm import PFMLoader
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import ImageOps

import utils as defs
import visualize as viz
from utils import get_depth_dir, get_img_dir, get_test_dir, tensor_imshow
from PIL import Image
import os
from os.path import join
from matplotlib import pyplot as plt
from visualize import blend_images
from data_loader import GeoposeDataset, GeoposeToTensor


class ResizeToImgShape:
    def __call__(self, sample):
        """
        resize every meta-image (like depth, segmentation, etc.)
        to be same resolution as main image.
        Returns: sample

        Args:
            sample: dict(Tensors), data sample with an 'image', 'depth' and possibly other images
        """
        _, h, w = sample['image'].shape
        for k, img in sample.items():
            if k == 'image':
                continue
            if torch.is_tensor(img):
                # using Nearest, bc we want values to stay mostly the same,
                # and it's worse to have -.8 than some out of place -1's (for sky values).
                # segmap really can't be interpolated, bc values are supposed to be constant.
                img = TF.resize(img, (h, w), interpolation=Image.NEAREST)
                tensor_imshow(img)
                sample[k] = img

        # blended = blend_images(TF.to_pil_image(image), TF.to_pil_image(depth_resized))
        # plt.imshow(blended)
        # plt.show()
        return sample


class CropToAspectRatio:
    """
    crop sample images to required aspect ratio.
    the image then can just be rescaled to the required resolution.
    assumes all tensors are 3d, and grayscale images are of shape 1xHxW.
    """

    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def __call__(self, sample):
        for k, img in sample.items():
            if torch.is_tensor(img):
                # tensor_imshow(img)
                h, w = img.shape[1], img.shape[2]
                if h * self.aspect_ratio > w:
                    disparity = int(h - w // self.aspect_ratio)
                    top = disparity // 2
                    height = h - disparity
                    img = TF.crop(img, top=top, left=0, height=height, width=w)
                    # tensor_imshow(img)
                    assert img.shape[1] * self.aspect_ratio == float(w), 'aspect ratio still not checking out.'
                    sample[k] = img
                elif h * self.aspect_ratio < w:
                    disparity = int(w - h * self.aspect_ratio)
                    left = disparity // 2
                    width = (w - ((disparity + 1) // 2 * 2))  # //2*2 to fix possible misalignment.
                    img = TF.crop(img, top=0, left=left, height=h, width=width)
                    # tensor_imshow(img)
                    assert img.shape[1] * self.aspect_ratio == float(w), 'aspect ratio still not checking out.'
                    sample[k] = img
        return sample


class ResizeToResolution:
    """
    resize all images in sample to be set resolution.
    Interpolation:
        - 'image' is resized using interpolation
        - all other images just fill with nearest value.
    to be used after cropping to aspect ratio with CropToAspectRatio.
    """

    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, sample):
        for k, img in sample.items():
            if torch.is_tensor(img):
                # tensor_imshow(img)
                if k == 'image':
                    sample[k] = TF.resize(img, (self.h, self.w))
                else:
                    sample[k] = TF.resize(img, (self.h, self.w),
                                          interpolation=Image.NEAREST)  # don't want -1's and shit to get f'd up.
                    # TODO: assert that values of images didn't change.
                # tensor_imshow(sample[k])
        return sample


class PadToResolution:
    """
    Pad all images in sample to meet required resolution. Pretty simple.
    """

    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, sample):
        for k, img in sample.items():
            if torch.is_tensor(img):
                # tensor_imshow(img)
                sample_h, sample_w = img.shape[1], img.shape[2]
                pad_h, pad_w = 0, 0
                if sample_h < self.h:
                    pad_h = self.h - sample_h
                if sample_w < self.w:
                    pad_w = self.w - sample_w
                # pad to get to required resolution,
                # add 1 extra pad for right and bottom if resolution diff is odd.
                sample[k] = TF.pad(img, (pad_w // 2, pad_h // 2, (pad_w + 1) // 2, (pad_h + 1) // 2))
                # tensor_imshow(sample[k])
        return sample


class CenterAndCrop:
    def __init__(self, height, width):
        """
        crop out center of image to be in resolution hxw.

        Args:
            height: height (x)
            width: width (y)
        """
        self.h = height
        self.w = width

    def __call__(self, sample):
        for k, v in sample.items():
            if torch.is_tensor(v):
                # tensor_imshow(v)
                sample[k] = TF.center_crop(v, (self.h, self.w))
                # tensor_imshow(sample[k])
        return sample


class ExtractSkyMask:
    def __call__(self, sample):
        depth = sample['depth']
        skymask = depth[depth == -1]
        sample['skymask'] = skymask
        return sample


class ExtractSegmentationMask:
    pass
    # TODO: next week, the whole segmentation git.


if __name__ == '__main__':
    # geoset = GeoposeDataset(transform=transforms.Compose([GeoposeToTensor(),
    #                                                       ResizeDepth(),
    #                                                       CenterAndCrop(680, 1024)
    #                                                       ]))
    # geoset_no_crop = GeoposeDataset(transform=transforms.Compose([GeoposeToTensor(),
    #                                                               ResizeDepth()]))
    # dl = DataLoader(geoset, batch_size=4)
    # batch = next(iter(dl))
    # s2 = next(iter(geoset_no_crop))
    # print single sample,
    # since if we don't crop we can't have them all in one batch.
    h, w = 680, 1020
    aspect_ratio = 3 / 2
    pad_and_center = transforms.Compose([GeoposeToTensor(),
                                         ResizeToImgShape(),
                                         PadToResolution(h, w),
                                         CenterAndCrop(h, w)])
    crop_to_aspect_ratio_and_resize = transforms.Compose([GeoposeToTensor(),
                                                          CropToAspectRatio(aspect_ratio=aspect_ratio),
                                                          ResizeToResolution(h, w),
                                                          ExtractSkyMask()])
    geoset = GeoposeDataset(transform=crop_to_aspect_ratio_and_resize)
    s2 = next(iter(geoset))
    for k, v in s2.items():
        if torch.is_tensor(v):
            s2[k] = v.unsqueeze(0)
        elif type(v).__name__ == 'str_':
            s2[k] = [v]
    # viz.show_batch(batch)
    # plt.show()
    viz.show_batch(s2)
    plt.show()
