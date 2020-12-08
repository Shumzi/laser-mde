from __future__ import print_function, division
import logging
import sys
from utils.definitions import DATA_DIR, get_depth_dir, get_img_dir
import utils.definitions as defs
import visualize as viz
# load and display an image with Matplotlib
import random
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils


class FarsightDataset(Dataset):
    """Farsight dataset with (img,depth) pairs."""

    def __init__(self, img_dir, depth_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            depth_dir (string): Directory with all the depth images
                  (with same name as its corresponding image).
            transform (callable, optional): Optional transform to be applied
                  on a sample.
        """
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.filenames = np.array([fn for fn in os.listdir(img_dir) if fn.lower().endswith('.png')])
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
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        dev = defs.get_cuda()
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in addition, conv2d doesn't really work with uint8,
        # so change to float32 & range [0..1] instead of [0..255].
        image = image.transpose((2, 0, 1)).astype(np.float32)/256
        # same problem with depth. depth is in bins of 4m, max 250 (1000m). Moving to [0..1] 
        depth = depth.astype(np.float32)/250
        # depth is just H X W, so no problem here.
        return {'image': torch.from_numpy(image).to(device=dev),
                'depth': torch.from_numpy(depth).to(device=dev),
                'name': sample['name']}


    
if __name__ == '__main__':
    dataloader = DataLoader(FarsightDataset(img_dir=get_img_dir(),
                                            depth_dir=get_depth_dir(),
                                            transform=ToTensor()),
                            batch_size=4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print('batch #{}, img size: {}, depth size: {}'.format(i_batch,
                                                               sample_batched['image'].size(),
                                                               sample_batched['depth'].size()))

        # observe 4th batch and stop.
        if i_batch == 0:
            fig = plt.figure()
            viz.show_batch(sample_batched)
            plt.title('hi')
            # plt.axis('off')
            plt.show()
            break

