import random
from random import shuffle

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision import transforms
from torchvision.transforms import functional as TF, Compose
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
import utils as defs
import visualize as viz
from data_loader import GeoposeDataset, GeoposeToTensor, FarsightDataset, FarsightToTensor
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
# segmentation imports
import os, csv, scipy.io
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from torch.nn import functional as F

from utils import cfg, get_img_dir
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
cfg_aug = defs.cfg['data_augmentation']


class ResizeToImgShape:
    def __call__(self, sample):
        """
        resize every meta-image (like depth, segmentation, etc.)
        to be same resolution as main image.
        Returns: sample

        Args:
            sample: dict(Tensors), data sample with an 'image', 'depth' and possibly other images
        """
        dims = {}
        for k, v in sample.items():
            if k != 'name':
                dims[k] = len(v.shape)
        d1, d2, d3 = sample['image'].shape
        for k, img in sample.items():
            if k == 'image':
                continue
            if torch.is_tensor(img) or isinstance(img, Image.Image):
                # using Nearest, bc we want values to stay mostly the same,
                # and it's worse to have -.8 than some out of place -1's (for sky values).
                # segmap really can't be interpolated, bc values are supposed to be constant.
                img = TF.resize(img, (d2, d3), interpolation=Image.NEAREST)
                sample[k] = img
            elif type(img) == np.ndarray:
                img = cv2.resize(img, (d2, d1), interpolation=cv2.INTER_NEAREST)
                sample[k] = img
        for k, v in sample.items():
            if k != 'name':
                assert dims[k] == len(v.shape), f"{k}'s dim changed from {dims[k]} to {len(v.shape)}"
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
                h, w = img.shape[1], img.shape[2]
                if h * self.aspect_ratio > w:
                    disparity = int(h - w // self.aspect_ratio)
                    top = disparity // 2
                    height = h - disparity
                    img = TF.crop(img, top=top, left=0, height=height, width=w)
                    sample[k] = img
                elif h * self.aspect_ratio < w:
                    disparity = int(w - h * self.aspect_ratio)
                    left = disparity // 2
                    width = w - disparity
                    img = TF.crop(img, top=0, left=left, height=h, width=width)
                    h, w = img.shape[1], img.shape[2]
                    assert int(h * self.aspect_ratio) == w, 'aspect ratio still not checking out.'
                    sample[k] = img
        return sample


def add_mask_to_img_if_in_configs(sample):
    """
    adds mask to image if so defined in configs
    Args:
        sample:

    Returns: added mask.

    """
    if defs.cfg['dataset']['use_mask'] and defs.cfg['dataset']['add_mask_to_image']:
        assert 'mask' in sample, 'use_mask set to True but no mask provided in sample.'
        mask = sample['mask']
        sample['image'] = torch.cat((sample['image'], mask), dim=0)
    return sample


class ResizeToAlmostResolution:
    """
    resize image to closest resolution without stretching image.
    """

    def __init__(self, height, width, upper_bound=False):
        """
        settings for resizing
        Args:
            height:
            width:
            upper_bound: use upper bound of image (for cropping).
        """
        self.h = height
        self.w = width
        self.ar = self.w / self.h
        self.upper_bound = upper_bound

    def __call__(self, sample):
        for k, img in sample.items():
            if isinstance(img, np.ndarray):
                h, w = img.shape[0], img.shape[1]
                ar = w / h
                if (self.ar > ar and not self.upper_bound) or (self.ar < ar and self.upper_bound):
                    new_h, new_w = self.h, int(self.h * ar)
                elif (self.ar < ar and not self.upper_bound) or (self.ar > ar and self.upper_bound):
                    new_h, new_w = int(self.w / ar), self.w
                else:
                    new_h, new_w = self.h, self.w
                if k == 'image':
                    sample[k] = np.array(
                        TF.resize(Image.fromarray(img), (new_h, new_w), interpolation=Image.BILINEAR))
                else:
                    sample[k] = cv2.resize(img, (new_w, new_h),
                                           interpolation=cv2.INTER_NEAREST)  # don't want -1's and shit to get f'd up.
            elif torch.is_tensor(img):
                # viz.tensor_imshow(img)
                if k == 'image':
                    sample[k] = F.interpolate(img.unsqueeze(0), (self.h, self.w),
                                              mode='bilinear', align_corners=False).squeeze(0)
                else:
                    sample[k] = TF.resize(img, (self.h, self.w),
                                          interpolation=Image.NEAREST)  # don't want -1's and shit to get f'd up.
                    # TODO: assert that values of images didn't change.
                # viz.tensor_imshow(sample[k])
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
            if isinstance(img, np.ndarray):
                if k == 'image':
                    sample[k] = np.array(
                        TF.resize(Image.fromarray(img), (self.h, self.w), interpolation=Image.BILINEAR))
                else:
                    sample[k] = cv2.resize(img, (self.w, self.h),
                                           interpolation=cv2.INTER_NEAREST)  # don't want -1's and shit to get f'd up.
            elif torch.is_tensor(img):
                # viz.tensor_imshow(img)
                if k == 'image':

                    sample[k] = F.interpolate(img.unsqueeze(0), (self.h, self.w),
                                              mode='bilinear', align_corners=False).squeeze(0)
                else:
                    sample[k] = TF.resize(img, (self.h, self.w),
                                          interpolation=Image.NEAREST)  # don't want -1's and shit to get f'd up.
                    # TODO: assert that values of images didn't change.
                # viz.tensor_imshow(sample[k])
        return sample


class PadToAspectRatio:
    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def __call__(self, sample):
        for k, img in sample.items():
            if torch.is_tensor(img) or isinstance(img, np.ndarray):
                # TODO: testing to see all images were originally in same resolution.
                if isinstance(img, np.ndarray):
                    sample_h, sample_w = img.shape[0], img.shape[1]
                else:
                    sample_h, sample_w = img.shape[1], img.shape[2]
                sample_aspect_ratio = sample_w / sample_h
                pad_h, pad_w = 0, 0
                if sample_aspect_ratio < self.aspect_ratio:
                    ar_disparity = self.aspect_ratio - sample_aspect_ratio
                    pad_w = ar_disparity * sample_h
                elif sample_aspect_ratio > self.aspect_ratio:
                    ar_disparity = sample_aspect_ratio - self.aspect_ratio
                    pad_h = ar_disparity * sample_h
                # add 1 extra pad for right and bottom if resolution diff is odd.
                border = (int(pad_w / 2), int(pad_h / 2), int((pad_w + 1) / 2), int((pad_h + 1) / 2))
                if isinstance(img, np.ndarray):
                    #  top, bottom, left, right
                    sample[k] = np.array(TF.pad(img, border))
                else:
                    sample[k] = TF.pad(img, border)
                    # sample[k] = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
                # else:
                # left, top, right and bottom

        return sample


class PadToResolution:
    """
    Pad all images in sample to meet required resolution. Pretty simple.
    """

    def __init__(self, height, width):
        self.pad_to_aspect_ratio = PadToAspectRatio(width / height)

    def __call__(self, sample):
        self.pad_to_aspect_ratio = PadToAspectRatio(sample)


class CenterCrop:
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
                # viz.tensor_imshow(v)
                sample[k] = TF.center_crop(v, (self.h, self.w))
                # viz.tensor_imshow(sample[k])
                assert sample[k].shape[1:3] == (self.h, self.w), f"problem with centering for {sample['name']}"
        return sample


class ExtractSkyMask:
    def __call__(self, sample):
        """
        Objective: mask out the sky (as it just confuses the model).
        In any case, we don't care about it.

        Args:
            sample: Geopose dataset sample, possibly already containing a 'mask'.

        Returns: sample with masked out sky added in 'mask' key (i.e.
                (added to existing mask, if exists).
        """
        depth = sample['depth']
        # (-1 - cfg['normalization']['depth_mean'] / cfg['normalization']['depth_std'])
        if depth.min() == -1:
            sky_mask = depth != -1
        elif depth.min() == 0:
            sky_mask = depth != 0
        else:
            sky_mask = torch.full_like(depth, True, dtype=torch.bool)

        if 'mask' in sample:
            current_mask = sample['mask']
            sample['mask'] = current_mask & sky_mask
        else:
            sample['mask'] = sky_mask
        assert sample['mask'].dtype == torch.bool, 'mask contains non-binary values'
        return sample


class ExtractSegmentationMask:
    def __init__(self):
        """
        extract semantic map from pretrained model, for knowing where to ignore in the image,
        since we only have depth info on mountains.
        """
        self.names = {}
        with open('semseg/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

        # Network Builders
        self.net_encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='semseg/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
        self.net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=2048,
            num_class=150,
            weights='semseg/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
            use_softmax=True)

        self.crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(self.net_encoder, self.net_decoder, self.crit)
        self.segmentation_module.eval()
        self.segmentation_module.to(device=defs.get_dev())

    def visualize_result(self, img, pred, index=None):
        # filter prediction class if requested
        colors = scipy.io.loadmat('semseg/color150.mat')['colors']
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
            print(f'{self.names[index + 1]}:')

        # colorize prediction
        pred_color = colorEncode(pred, colors).astype(np.float32) / 255

        # aggregate images and save
        im_vis = np.concatenate((img, pred_color), axis=1)
        plt.imshow(im_vis)
        plt.show()

    def __call__(self, sample):
        img_data = sample['image']
        old_shape = img_data.shape[1], img_data.shape[2]
        img_data = ResizeToAlmostResolution(180, 224)({'image': img_data})['image']
        singleton_batch = {'img_data': img_data[None]}
        output_size = img_data.shape[1:]
        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch, segSize=output_size)
        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = TF.resize(pred, old_shape, Image.NEAREST)
        # other irrelevant classes: 1, 4, 12, 20, 25, 83, 116, 126, 127.
        # see csv in semseg folder.
        bad_classes = torch.Tensor([2]).to(device=defs.get_dev())
        mask = torch.full_like(pred, True, dtype=torch.bool)
        mask[(pred[..., None] == bad_classes).any(-1)] = False
        if 'mask' in sample:
            sample['mask'] = sample['mask'] & mask
        else:
            sample['mask'] = mask
        return sample


class ExtractSegmentationMaskSimple:
    """same idea as non-simple counterpart, only uses a simpler model.
    Categories are as follows (we only care about background):
    {0: '__background__',
     1: 'aeroplane',
     2: 'bicycle',
     3: 'bird',
     4: 'boat',
     5: 'bottle',
     6: 'bus',
     7: 'car',
     8: 'cat',
     9: 'chair',
     10: 'cow',
     11: 'diningtable',
     12: 'dog',
     13: 'horse',
     14: 'motorbike',
     15: 'person',
     16: 'pottedplant',
     17: 'sheep',
     18: 'sofa',
     19: 'train'}
"""

    def __init__(self):
        self.segmodel = deeplabv3_resnet50(pretrained=True).to(device=defs.get_dev())
        self.segmodel.eval()

    def __call__(self, sample):
        segmap = self.segmodel(sample['image'].unsqueeze(0))['out'][0]
        segmap_pred = segmap.argmax(0)
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        # plot the semantic segmentation predictions of 21 classes in each color
        h, w = list(sample['image'].shape[1:])
        segmentation_mask = segmap_pred == 0
        if 'mask' in sample:
            sample['mask'] = sample['mask'] & segmentation_mask
        else:
            sample['mask'] = segmentation_mask
        return sample


class ReverseMask:
    def __call__(self, sample):
        sample['mask'] = torch.logical_not(sample['mask'])
        return sample


class DepthToZScore:
    def __call__(self, sample):
        depth_mean = torch.Tensor([defs.cfg['normalization']['depth_mean']]).to(device=defs.get_dev())
        depth_std = torch.Tensor([defs.cfg['normalization']['std_mean']]).to(device=defs.get_dev())
        sample['depth'] = TF.normalize(sample['depth'], depth_mean, depth_std).to(device=defs.get_dev())
        # adding min to norm so all values will be above 0 (if we're using rmsle).
        if defs.cfg['optim']['loss'].lower() == 'rmsle':
            sample['depth'] += torch.Tensor([(depth_mean + 1) / depth_std]).to(device=defs.get_dev())
        return sample


def norm_img_imagenet(sample):
    sample['og_image'] = sample['image'].clone()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    sample['image'] = TF.normalize(sample['image'], mean=mean, std=std)
    return sample


class NormalizeImg:
    """normalize images and depths based on our crop results."""

    def __call__(self, sample):
        image_mean = torch.Tensor(defs.cfg['normalization']['image_mean']).to(defs.get_dev())
        image_std = torch.Tensor(defs.cfg['normalization']['image_std']).to(defs.get_dev())
        sample['image'] = TF.normalize(sample['image'], image_mean, image_std)
        return sample


class NormMinMaxDepth:
    """
    minmax scaling, see
    https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    """

    def __call__(self, sample):
        # epsilon = 1e-10
        min_val = defs.cfg['normalization']['depth_min']
        max_val = defs.cfg['normalization']['depth_max']  # found from EDA,
        depth = sample['depth']
        depth -= min_val
        depth /= max_val - min_val
        depth = torch.clamp(depth, max=1)
        # depth += epsilon  # for ability to plot.
        assert depth.min() >= 0 and depth.max() <= 1, "depth is out of [0-1] range"
        sample['depth'] = depth
        return sample


class reverseMinMaxScale():
    def __call__(self, sample):
        """
        reverses the minmax scaling done to the depth image.
        Args:
            img: depth image to be rescaled to original depths

        Returns: rescaled image.

        """
        min_val = defs.cfg['normalization']['depth_min']
        max_val = defs.cfg['normalization']['depth_max']  # found from EDA,
        sample['depth'] *= max_val - min_val
        sample['pred'] *= max_val - min_val
        sample['depth'] += min_val
        sample['pred'] += min_val
        return sample


def pad_to_aspect_ratio_and_resize(dataset_name):
    cfg_ds = defs.cfg['dataset']
    aspect_ratio, h, w = cfg_ds['aspect_ratio'], cfg_ds['h'], cfg_ds['w']
    tf = transforms.Compose([FillNaNsFFS(),
                             DatasetToTensor(dataset_name),
                             PadToAspectRatio(aspect_ratio=aspect_ratio),
                             ResizeToResolution(height=h, width=w),
                             norm_img_imagenet])
    return Compose([tf, normalize_depth(dataset_name)])


def normalize_depth(dataset='geopose'):
    """
    Args:
        dataset:

    Returns: normalization object for dataset according to configs.yml.

    """
    if dataset == 'farsight':
        norm = Compose([])
    elif defs.cfg['normalization']['depth_norm'] == 'minmax':
        norm = NormMinMaxDepth()
    elif defs.cfg['normalization']['depth_norm'] == 'z':
        norm = DepthToZScore()
    return norm


class RandomHorizontalFlip:
    """
    flips image and its depth with probability p.
    """

    def __init__(self, p=cfg_aug['flip_p']):
        self.p = p

    def __call__(self, sample):
        if cfg_aug['horizontal_flip'] and random.random() < self.p:
            for k, v in sample.items():
                if torch.is_tensor(v):
                    sample[k] = TF.hflip(v)
        return sample


class RandomColorJitter:
    """
    jitter colors of img, but not depth.
    """

    def __call__(self, sample):
        if cfg_aug['color_jitter']:
            sample['image'] = transforms.ColorJitter().forward(sample['image'])
        return sample


class RandomGaussianBlur:
    """
    blur image with small gaussian blur (depth stays the same).
    """

    def __init__(self, p=0.5):
        """
        initialize probability with which to add blur.
        Args:
            p: float, probability for blurring.
        """
        self.p = p

    def __call__(self, sample):
        if cfg_aug['gaussian_blur'] and random.random() < self.p:
            sample['image'] = transforms.GaussianBlur(3).forward(sample['image'])
        return sample


class RandomGaussianNoise:
    def __init__(self, p=0.5, std=0.01):
        self.p = p
        self.std = std

    def __call__(self, sample):
        if cfg_aug['gaussian_noise'] and random.random() < self.p:
            sample['image'] = sample['image'] + torch.randn(sample['image'].size(), device=defs.get_dev()) * self.std
            sample['image'] = torch.clamp(sample['image'], 0, 1)
        return sample


def get_augmentations():
    """
    Returns: composition of image augmentations, according to configs.yml.

    """
    tforms = Compose([])
    if cfg_aug['horizontal_flip']:
        tforms = Compose([tforms, RandomHorizontalFlip()])
    if cfg_aug['color_jitter']:
        tforms = Compose([tforms, RandomColorJitter()])
    if cfg_aug['gaussian_blur']:
        tforms = Compose([tforms, RandomGaussianBlur()])
    if cfg_aug['gaussian_noise']:
        tforms = Compose([tforms, RandomGaussianNoise()])
    return tforms


class DatasetToTensor:
    """
    Generalizing class for <DatasetName>ToTensor
    """

    def __init__(self, dataset):
        if dataset == 'geopose':
            self.totensor = GeoposeToTensor()
        elif dataset == 'farsight':
            self.totensor = FarsightToTensor()

    def __call__(self, sample):
        return self.totensor(sample)


def resize_and_crop_center(dataset):
    cfg_ds = defs.cfg['dataset']
    h, w = cfg_ds['h'], cfg_ds['w']
    cmp = transforms.Compose([FillNaNsFFS(),
                              ResizeToImgShape(),
                              ResizeToAlmostResolution(h, w, upper_bound=True),
                              DatasetToTensor(dataset),
                              CenterCrop(h, w),
                              norm_img_imagenet,
                              ])
    return cmp


class ToPIL:
    def __call__(self, sample):
        for k, img in sample.items():
            if isinstance(img, np.ndarray):
                sample[k] = Image.fromarray(img)
        return sample


class FillNaNsFFS:
    """fills in possible nans in a depth map."""

    def __call__(self, sample):
        depth = sample['depth']
        sample['depth'] = np.nan_to_num(depth)
        return sample


def get_geopose_split(val_percent, test_percent, subset_size=None):
    """
    splitting non-randomly bc some images that are close in folder are similar in content.
    So, to eliminate data-leakage + since all other images are dissimilar anyway, just doing constant split.
    Args:
        val_percent: percentage from total dataset for validation
        test_percent: percentage from total dataset for test.
        subset_size: int, size of subset from geopose dataset to use in total. default: everything.

    Returns: train_split, val_split, test_split: Datasets of train,val and test.

    """
    tf = get_transform()
    ds = GeoposeDataset(transform=tf)
    if subset_size is not None:
        ds = Subset(ds, range(subset_size))
    n_test = int(len(ds) * test_percent)
    n_val = int(len(ds) * val_percent)
    n_train = len(ds) - (n_val + n_test)
    logger.info(f'\nds size: {len(ds)}'
                f'\ntrain size: {n_train}'
                f'\nval_size: {n_val}'
                f'\ntest_size (might be irrelevant): {n_test}')
    train_split = Subset(ds, range(n_train))
    val_split = Subset(ds, range(n_train, n_train + n_val))
    test_split = Subset(ds, range(n_train + n_val, n_train + n_val + n_test))
    assert len(train_split) + len(val_split) + len(test_split) == len(ds), \
        'sizes of datasets don\'t add up to original dataset.' \
        f'\n{len(train_split) + len(val_split) + len(test_split)} vs {len(ds)}'
    return train_split, val_split, test_split


def get_transform(dataset_name='geopose'):
    cfg_dataset = cfg['dataset']
    cropping_system = cfg_dataset['cropping_system']
    if cropping_system == 'resize_and_crop_center':
        tf = resize_and_crop_center(dataset_name)
    elif cropping_system == 'pad_to_aspect_ratio_and_resize':
        tf = pad_to_aspect_ratio_and_resize(dataset_name)
    tf = Compose(
        [tf, normalize_depth(dataset_name), get_mask_transform(), get_augmentations(), add_mask_to_img_if_in_configs])
    return tf


def get_mask_transform():
    cfg_dataset = cfg['dataset']
    if cfg_dataset['use_mask']:
        tf = ExtractSkyMask()
        if cfg_dataset['mask_type'] == 'simple':
            tf = Compose([tf, ExtractSegmentationMaskSimple()])
        elif cfg_dataset['mask_type'] == 'smart':
            tf = Compose([tf, ExtractSegmentationMask()])
        elif cfg_dataset['mask_type'] == 'sky_only':
            pass
        else:
            raise Exception('didn\'t specify a legal mask_type but requested to use mask')
        if cfg_dataset['reverse_mask']:
            tf = Compose([tf, ReverseMask()])
    else:
        tf = Compose([])
    return tf


def get_mixed_datasets(val_percent, test_percent, subset_size=None):
    """
    returns a mixed Dataset comprised of the geopose and farsight datasets.
    Args:
        val_percent: used by both datasets
        test_percent: percentage of geopose to use for test.
        subset_size: (default: everything.)
    Returns: train_ds, val_ds, geo_test.

    """
    geo_train, geo_val, geo_test = get_geopose_split(val_percent, test_percent, subset_size)
    farsight_train, farsight_val = get_farsight_split(subset_size, val_percent)
    train_ds = ConcatDataset([geo_train, farsight_train])
    val_ds = ConcatDataset([geo_val, farsight_val])
    return train_ds, val_ds, geo_test


def get_loaders():
    """
    get data loaders for train set and val set
    TODO: make reverseDepth be separate from get_loaders.
    Returns: dataloaders + possibly reverse function for depth, all according to configs.

    """
    cfg_train = cfg['train']
    cfg_validation = cfg['validation']
    batch_size = cfg_train['batch_size']
    batch_size_val = cfg['validation']['batch_size']
    ds_name = cfg['dataset']['name']
    subset_size = cfg_train['subset_size']
    val_percent = cfg_validation['val_percent']
    test_percent = cfg_validation['test_percent']
    if not val_percent:
        val_percent = 0
    if ds_name == 'geopose_and_farsight':
        train_split, val_split, test_split = get_mixed_datasets(val_percent, test_percent, subset_size)
    if ds_name == 'farsight':
        train_split, val_split = get_farsight_split(subset_size, val_percent)
    elif ds_name == 'geopose':
        train_split, val_split, test_split, = get_geopose_split(val_percent=val_percent,
                                                                test_percent=test_percent,
                                                                subset_size=subset_size)
    loaders = []
    train_loader = DataLoader(train_split,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=0)
    loaders.append(train_loader)
    if cfg_validation['val_round']:
        val_loader = DataLoader(val_split,
                                shuffle=cfg_validation['shuffle_val'],
                                batch_size=batch_size_val,
                                num_workers=0)
        loaders.append(val_loader)
    if cfg_validation['get_test']:
        test_loader = DataLoader(test_split,
                                 shuffle=False,
                                 batch_size=batch_size_val,
                                 num_workers=0)
        loaders.append(test_loader)
        loaders.append(reverseMinMaxScale)
    return loaders


def get_farsight_split(subset_size, val_percent):
    cfg_train = cfg['train']
    if cfg_train['use_folds']:
        train_split, val_split = get_farsight_fold_dataset(1)
        if subset_size is not None:
            train_size = int(subset_size * (1 - val_percent))
            val_size = int(subset_size * val_percent)
            train_split = Subset(train_split, range(train_size))
            val_split = Subset(val_split, range(val_size))
    else:
        ds = FarsightDataset(transform=get_transform(dataset_name='farsight'))
        if subset_size is not None:
            ds = Subset(ds, range(subset_size))
        n_val = int(len(ds) * val_percent)
        n_train = len(ds) - n_val
        train_split, val_split = random_split(ds,
                                              [n_train, n_val],
                                              generator=torch.Generator().manual_seed(42))
    return train_split, val_split


def get_farsight_fold_dataset(fold):
    """
    creates a train and val dataset,
    with the fold being which scene is used as val.

    Args:
        fold: index, scene no. to be used as val.

    Returns: train_dataset, val_dataset

    """
    assert fold < 4, "Farsight has only 4 scenes, fold must be between 0 and 3."
    files = [fn for fn in os.listdir(get_img_dir()) if fn.lower().endswith('.png')]
    files.sort()
    files_df = pd.DataFrame(files, columns=['filename'])
    files_df['city'] = files_df['filename'].apply(lambda x: x.split(sep='_')[0])
    train_idxs = []
    val_idxs = []
    for i, (_, g) in enumerate(files_df.groupby('city')):
        if i == fold:
            val_idxs += list(g['filename'].values)
        else:
            train_idxs += list(g['filename'].values)
    shuffle(train_idxs)
    # TODO: make this not so shady as it is now. Maybe it's ok?
    if defs.cfg['validation']['shuffle_val']:
        shuffle(val_idxs)
    train_ds = FarsightDataset(get_transform(dataset_name='farsight'), train_idxs)
    val_ds = FarsightDataset(get_transform(dataset_name='farsight'), val_idxs)
    return train_ds, val_ds


if __name__ == '__main__':
    import logging

    loaders = get_loaders()
    train_loader = loaders[0]
    viz.show_batch(next(iter(train_loader)))

    print('done')
