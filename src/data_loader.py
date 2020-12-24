from __future__ import print_function, division
import logging
import sys

from utils import get_depth_dir, get_img_dir
import utils as defs
import visualize as viz
# load and display an image with Matplotlib
import random
import os
import torch
# import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from itertools import groupby
import pandas as pd


class FarsightDataset(Dataset):
    """Farsight dataset with (img,depth) pairs."""

    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                  on a sample.
        """
        self.img_dir = get_img_dir()
        self.depth_dir = get_depth_dir()
        self.transform = transform
        self.filenames = np.array([fn for fn in os.listdir(self.img_dir) if fn.lower().endswith('.png')])
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
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img, depth = sample['image'], sample['depth']
            img = TF.hflip(img)
            depth = TF.hflip(depth)
            sample['image'] = img
            sample['depth'] = depth
        return sample


class RandomColorJitter:
    """
    jitter colors of img, but not depth.
    """

    def __call__(self, sample):
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
        if random.random() < self.p:
            sample['image'] = transforms.GaussianBlur(3).forward(sample['image'])
        return sample

class RandomGaussianNoise:
    pass #TODO: rndgaussiannoise.


def get_farsight_fold_dataset(fold, transform=ToTensor()):
    """
    creates a train and val dataset,
    with the fold being which scene is used as val.
    Args:
        fold: int, scene no. to be used as val.
        transform: transform object for dataset.
    Returns: train_dataset, val_dataset

    """
    assert fold < 4, "Farsight has only 4 scenes, fold must be between 0 and 3."
    files = [fn for fn in os.listdir(get_img_dir()) if fn.lower().endswith('.png')]
    files.sort()
    files_df = pd.DataFrame(files, columns=['filename'])
    files_df['city'] = files_df['filename'].apply(lambda x: x.split(sep='_')[0])
    full_ds = FarsightDataset(transform)
    train_idxs = []
    val_idxs = []
    for i, (_, g) in enumerate(files_df.groupby('city')):
        if i == fold:
            val_idxs += list(g.index.values)
        else:
            train_idxs += list(g.index.values)
    val_ds = Subset(full_ds, val_idxs)
    train_ds = Subset(full_ds, train_idxs)
    return train_ds, val_ds


# transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

# transforms.RandomHorizontalFlip(p=0.5)

# transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))


if __name__ == '__main__':
    """
    basic test to see that the dataloader works ok.
    """
    tforms = transforms.Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        RandomColorJitter(),
        RandomGaussianBlur()
    ])
    t, v = get_farsight_fold_dataset(0, tforms)
    t_loader = DataLoader(t, batch_size=4)
    v_loader = DataLoader(v, batch_size=4)
    for i_batch, sample_batched in enumerate(t_loader):
        viz.show_batch(sample_batched)
        plt.title('train')
        plt.show()
        break
    for batch in v_loader:
        viz.show_batch(batch)
        plt.title('test')
        plt.show()
        break
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
