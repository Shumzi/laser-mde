import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import visualize as viz
from data_loader import FarsightTestDataset, GeoposeDataset
from train import load_checkpoint
from utils import get_dev
from prepare_data import ResizeToAlmostResolution, GeoposeToTensor, CenterAndCrop, norm_img_imagenet
from torchvision.transforms import Compose
from utils import cfg
"""
assuming model works, just get images and output depth maps for said images.
"""


def single_image_to_tensor(image):
    image = (image.transpose((2, 0, 1)).astype(np.float32) / 256)[:, :-1, :-1]
    torch_image = torch.from_numpy(image).to(device=get_dev())
    return torch_image


def infer():
    net, optim, epoch, loss = load_checkpoint()
    net.eval()
    dataset = cfg['dataset']['name']
    h, w = cfg['dataset']['h'], cfg['dataset']['w']
    if dataset.lower() == 'geopose':
        compose = Compose([ResizeToAlmostResolution(h, w, upper_bound=True),
                           GeoposeToTensor(),
                           CenterAndCrop(h, w),
                           norm_img_imagenet, ])
        # test_ds = DataLoader(GeoposeDataset(transform=))

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
