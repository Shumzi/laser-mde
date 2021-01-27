"""
assuming model works, just get images and output depth maps for said images.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import visualize as viz
from data_loader import FarsightTestDataset
from train import load_checkpoint
from utils import get_dev


def single_image_to_tensor(image):
    image = (image.transpose((2, 0, 1)).astype(np.float32) / 256)[:, :-1, :-1]
    torch_image = torch.from_numpy(image).to(device=get_dev())
    return torch_image


def infer():
    net, optim, epoch, loss = load_checkpoint()
    net.eval()

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
    infer()
