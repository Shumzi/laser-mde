from __future__ import print_function, division
import logging
import sys

from utils import get_depth_dir, get_img_dir
import utils as defs
import visualize as viz
import random
import os
import torch
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
import pandas as pd
from random import shuffle

cfg_aug = defs.cfg['train']['data_augmentation']
class FarsightDataset(Dataset):
    """Farsight dataset with (img,depth) pairs."""

    def __init__(self, transform=None, filenames=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                  on a sample.
        """
        self.img_dir = get_img_dir()
        self.depth_dir = get_depth_dir()
        self.transform = transform
        if filenames is None:
            self.filenames = np.array([fn for fn in os.listdir(self.img_dir) if fn.lower().endswith('.png')])
        else:
            self.filenames = filenames
        self.filenames.sort()

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
    shuffle(val_idxs)
    shuffle(train_idxs)
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
        v_sample = next(iter(v_loader))
        viz.show_batch(t_sample)
        plt.suptitle(f'train_{i}')
        plt.show()
        viz.show_batch(v_sample)
        plt.suptitle(f'val_{i}_{len(v_loader)}')
        plt.show()

    # dataloader = DataLoader(FarsightDataset(transform=ToTensor()),
    #                         batch_size=4, shuffle=True, num_workers=0)
    #
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print('batch #{}, img size: {}, depth size: {}'.format(i_batch,
    #                                                            sample_batched['image'].size(),
    #                                                            sample_batched['depth'].size()))
    #
    #     # observe 4th batch and stop.
    #     if i_batch == 3:
    #         fig = plt.figure()
    #         viz.show_batch({**sample_batched,
    #                         'disp': sample_batched['depth'] - sample_batched['depth']})
    #         plt.title('hi')
    #         # plt.axis('off')
    #         plt.show()
    #         break
