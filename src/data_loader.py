import os
from os.path import join

import numpy as np
import torch
from pypfm import PFMLoader
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

import utils as defs
from utils import get_depth_dir, get_img_dir, get_test_dir
import logging
from utils import cfg

logger = logging.getLogger(__name__)
if cfg['misc']['verbose']:
    logger.setLevel(logging.INFO)


class GeoposeDataset(Dataset):
    """GeoPose3K dataset with (img,depth) pairs."""

    def __init__(self, transform=None, onlydepth=False):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                  on a sample.
        """
        self.dir = defs.cfg['dataset']['geopose_path']
        self.transform = transform
        if transform is not None and defs.cfg['misc']['verbose']:
            logger.info('using the following transforms:')
            logger.info(self.transform)
        self.foldernames = np.array(
            [fn for fn in os.listdir(self.dir) if os.path.isdir(join(self.dir, fn)) and not fn.startswith('.')])
        self.foldernames.sort()
        self.onlydepth = onlydepth

    def __len__(self):
        return len(self.foldernames)

    def __getitem__(self, idx):
        """
        get sample item
        Args:
            idx: int or string. if int: will get by idx in foldernames list,
                                if string: will get sample by folder name supplied in string.

        Returns:

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) == str:
            # assume idx is folder name requested. for visualization purposes mostly.
            folder = os.path.join(self.dir, idx)
        else:
            folder = os.path.join(self.dir, self.foldernames[idx])
        depth_name = None
        segmap_name = None
        img_name = None
        loader = PFMLoader()
        for file in os.listdir(folder):
            if file == 'distance_crop.pfm':
                depth_name = os.path.join(folder, file)
            elif file.endswith(('.png', '.jpg', '.jpeg')):
                if file == 'labels_crop.png':
                    segmap_name = join(folder, file)
                elif file.startswith('photo'):
                    img_name = os.path.join(folder, file)
        if not depth_name or not segmap_name or not img_name:
            raise Exception(f'folder is missing image/depth/seg: {folder}')

        if self.onlydepth:
            return {'depth': np.array(loader.load_pfm(depth_name)[::-1])}
        # image = Image.open(img_name)
        image = np.array(io.imread(img_name))
        depth = np.array(loader.load_pfm(depth_name)[::-1])
        name = idx if type(idx) == str else self.foldernames[idx]
        sample = {'image': image, 'depth': depth, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class GeoposeToTensor:
    """
    simple transform of np images to tensor images.
    """
    def __call__(self, sample):
        for k, v in sample.items():
            if type(v) == np.ndarray:
                sample[k] = TF.to_tensor(v).to(device=defs.get_dev(), dtype=torch.float32)
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


class FarsightToTensor(object):
    """Convert ndarrays in sample to Tensors
    and moves them into the correct range for conv operations."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        dev = defs.get_dev()
        image = TF.to_tensor(image).to(device=dev)
        # depth has max value of 1km. Moving to [0..1]
        depth = TF.to_tensor(depth / 1000).to(device=dev, dtype=torch.float32)
        sample['image'], sample['depth'] = image, depth
        return sample


if __name__ == '__main__':
    """
    basic test to see that the dataloader works ok.
    """
    # for i in range(4):
    #     t, v = get_farsight_fold_dataset(i, tforms)
    #     t_loader = DataLoader(t, batch_size=4)
    #     v_loader = DataLoader(v, batch_size=4)
    #     t_sample = next(iter(t_loader))
    #     # v_sample = next(iter(v_loader))
    #     viz.show_batch(t_sample)
    #     plt.suptitle(f'train_{i}')
    #     plt.show()
    #     # viz.show_batch(v_sample)
    #     # plt.suptitle(f'val_{i}_{len(v_loader)}')
    #     # plt.show()
