from __future__ import print_function, division

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
from PIL import Image, ImageOps
import utils as defs
import visualize as viz
from utils import get_depth_dir, get_img_dir, get_test_dir

cfg_aug = defs.cfg['data_augmentation']


class CropToAspectRatio:
    """
    crop sides so as to get the required aspect ratio.
    the image then can just be rescaled to required resolution.
    """

    def __init__(self, aspect_ratio, xres, yres):
        self.aspect_ratio = aspect_ratio
        self.xres, self.yres = xres, yres

    def __call__(self, imgs: list[Image]):
        for img in imgs:
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


class GeoposeDataset(Dataset):
    """GeoPose3K dataset with (img,depth) pairs."""

    def __init__(self, transform=None):
        """

        Args:
            transform (callable, optional): Optional transform to be applied
                  on a sample.
        """
        self.dir = '/home/bina/PycharmProjects/laser-mde/data/geoPose3K'
        self.transform = transform
        self.foldernames = np.array(
            [fn for fn in os.listdir(self.dir) if os.path.isdir(join(self.dir, fn)) and not fn.startswith('.')])
        self.foldernames.sort()

    def __len__(self):
        return len(self.foldernames)

    def __getitem__(self, idx):
        # batch requests
        if torch.is_tensor(idx):
            idx = idx.tolist()
        folder = os.path.join(self.dir, self.foldernames[idx])
        img_names = []
        depth_names = []
        segmap_names = []
        loader = PFMLoader()

        # for folder in batch_folders:
        #     print(folder)
        for file in os.listdir(folder):
            if file == 'distance_crop.pfm':
                depth_name = os.path.join(folder, file)
                # depth_names.append(os.path.join(folder, file))
            if file.endswith(('.png', '.jpg', '.jpeg')):
                if file == 'labels_crop.png':
                    # segmap_names.append(join(folder, file))
                    segmap_name = join(folder, file)
                else:
                    # img_names.append(os.path.join(folder, file))
                    img_name = os.path.join(folder, file)
        if len(img_names) != len(depth_names) or len(segmap_names) != len(depth_names):
            raise Exception(f'don\'t have same amount of images/segmaps as depths.\n'
                            # f' imgs: {len(img_names)}\ndepths: {len(depth_names)}\n'
                            f'folder: {folder}')
        image = io.imread(img_name)
        depth = loader.load_pfm(depth_name)
        segmap = io.imread(segmap_name)
        # bad - all images are in different resolutions!!!
        sample = {'image': image, 'depth': depth, 'segmap': segmap, 'name': self.foldernames[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FarsightDataset(Dataset):
    """Farsight dataset with (img,depth) pairs."""

    def __init__(self, transform=None, filenames=None):
        """

        Args:
            transform (callable, optional): Optional transform to be applied
                  on a sample.
            filenames: (list of strings, optional):
                filenames in folders to use for dataset, instead of all files (default).
                Example usage would be when making two datasets (train and test),
                having to specify which files go where, since split must be by city.
        """
        self.img_dir = get_img_dir()
        self.depth_dir = get_depth_dir()
        self.transform = transform
        if filenames is None:
            self.filenames = np.array([fn for fn in os.listdir(self.img_dir) if fn.lower().endswith('.png')])
            self.filenames.sort()
        else:
            self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # batch requests
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir,
                                self.filenames[idx])
        depth_name = os.path.join(self.depth_dir,
                                  self.filenames[idx])
        image = io.imread(img_name)
        depth = io.imread(depth_name)
        sample = {'image': image, 'depth': depth, 'name': self.filenames[idx].strip('.png')}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FarsightTestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.test_dir = get_test_dir()
        self.filenames = np.array([fn for fn in os.listdir(self.test_dir) if fn.lower().endswith('.png')])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.test_dir,
                                self.filenames[idx])
        image = io.imread(img_name)
        if self.transform is not None:
            image = self.transform(image)

        return {'image': image, 'name': self.filenames[idx].strip('.png')}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors
    and moves them into the correct range for conv operations."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        dev = defs.get_dev()
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in addition, conv2d doesn't really work with uint8,
        # so change to float32 & range [0..1] instead of [0..255].
        # transpose conv works bad with 513x513 images, so take only 512x512 from farsight imgs.
        image = (image.transpose((2, 0, 1)).astype(np.float32) / 256)[:, :-1, :-1]
        # image = image*2-1
        # same problem with depth. depth is in bins of 4m, max 250 (1000m). Moving to [0..1] 
        depth = (depth.astype(np.float32) / 250)[:-1, :-1]
        # depth is just H X W, so no problem here.
        image_tensor = torch.from_numpy(image).to(device=dev)
        depth_tensor = torch.from_numpy(depth).to(device=dev)
        return {'image': image_tensor,
                'depth': depth_tensor,
                'name': sample['name']}


class RandomHorizontalFlip:
    """
    flips image and its depth with probability p.
    """

    def __init__(self, p=cfg_aug['flip_p']):
        self.p = p

    def __call__(self, sample):
        if cfg_aug['horizontal_flip'] and random.random() < self.p:
            sample['image'] = TF.hflip(sample['image'])
            sample['depth'] = TF.hflip(sample['depth'])
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


def get_farsight_fold_dataset(fold, transform=None):
    """
    creates a train and val dataset,
    with the fold being which scene is used as val.

    Args:
        fold: index, scene no. to be used as val.
        transform: transform object for dataset.

    Returns: train_dataset, val_dataset

    """
    if transform is None:
        transform = transforms.Compose([
            ToTensor(),
            RandomHorizontalFlip(),
            RandomColorJitter(),
            RandomGaussianBlur(),
            RandomGaussianNoise()
        ])
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
    train_ds = FarsightDataset(transform, train_idxs)
    val_ds = FarsightDataset(ToTensor(), val_idxs)
    return train_ds, val_ds


if __name__ == '__main__':
    """
    basic test to see that the dataloader works ok.
    """
    tforms = transforms.Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        RandomColorJitter(),
        RandomGaussianBlur(),
        RandomGaussianNoise()
    ])
    for i in range(4):
        t, v = get_farsight_fold_dataset(i, tforms)
        t_loader = DataLoader(t, batch_size=4)
        v_loader = DataLoader(v, batch_size=4)
        t_sample = next(iter(t_loader))
        # v_sample = next(iter(v_loader))
        viz.show_batch(t_sample)
        plt.suptitle(f'train_{i}')
        plt.show()
        # viz.show_batch(v_sample)
        # plt.suptitle(f'val_{i}_{len(v_loader)}')
        # plt.show()
