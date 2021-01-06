"""
assuming model works, just get images and output depth maps for said images.
"""
import sys

from torch.utils.data import DataLoader
import numpy as np
from data_loader import get_farsight_fold_dataset, FarsightTestDataset, ToTensor
from train import load_checkpoint
import torch
import visualize as viz
from skimage import io
from matplotlib import pyplot as plt
from utils import cfg, get_test_dir, get_dev
import os


def single_image_to_tensor(image):
    image = (image.transpose((2, 0, 1)).astype(np.float32) / 256)[:, :-1, :-1]
    torch_image = torch.from_numpy(image).to(device=get_dev())
    return torch_image


def infer(img_names):
    net, optim, epoch, loss = load_checkpoint()
    net.eval()
    images = []

    test_ds = DataLoader(FarsightTestDataset(transform=single_image_to_tensor))
    image_batch = None
    depth_batch = None
    names = []
    for sample in test_ds:
        image = sample['image']
        names += sample['name']
        pred = net(image)
        if image_batch is None:
            image_batch = image
            depth_batch = pred.unsqueeze(0)
        else:
            image_batch = torch.cat((image_batch, image), 0)
            depth_batch = torch.cat((depth_batch, pred.unsqueeze(0)), 0)
    fig = viz.show_batch({'image': image_batch, 'depth': depth_batch, 'name': names})
    fig.suptitle('predictions', fontsize='xx-large')
    plt.show()


if __name__ == "__main__":
    cfg_ds = cfg['dataset']
    path = get_test_dir()
    image_names = os.listdir(path)
    infer(image_names)
