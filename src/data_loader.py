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
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils


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


if __name__ == '__main__':
    """
    basic test to see that the dataloader works ok.
    """
    dataloader = DataLoader(FarsightDataset(transform=ToTensor()),
                            batch_size=4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print('batch #{}, img size: {}, depth size: {}'.format(i_batch,
                                                               sample_batched['image'].size(),
                                                               sample_batched['depth'].size()))

        # observe 4th batch and stop.
        if i_batch == 3:
            fig = plt.figure()
            viz.show_batch({**sample_batched,
                            'disp': sample_batched['depth']-sample_batched['depth']})
            plt.title('hi')
            # plt.axis('off')
            plt.show()
            break
