import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import io
from torchvision.transforms import Compose

import visualize as viz
from data_loader import GeoposeDataset
from depthestim import pred as ken_net
from prepare_data import ResizeToAlmostResolution, GeoposeToTensor, CenterCrop, norm_img_imagenet, get_transform, \
    get_loaders, reverseMinMaxScale
from utils import cfg
from utils import get_dev, load_checkpoint

"""
assuming model works, just get images and output depth maps for said images.
"""


def infer():
    net, optim, epoch, loss = load_checkpoint()
    net.eval()
    dataset = cfg['dataset']['name']
    h, w = cfg['dataset']['h'], cfg['dataset']['w']
    if dataset.lower() == 'geopose':
        compose = Compose([ResizeToAlmostResolution(h, w, upper_bound=True),
                           GeoposeToTensor(),
                           CenterCrop(h, w),
                           norm_img_imagenet, ])
        # test_ds = DataLoader(GeoposeDataset(transform=))
    loaders = get_loaders()
    test_loader = loaders[2]
    image_batch = None
    depth_batch = None
    names = []
    for sample in test_loader:
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

    is_ken = True
    if is_ken:
        net = ken_net
    else:
        net, _, _, _ = load_checkpoint()
        rescale = reverseMinMaxScale()
    img = np.array(io.imread('photo.jpeg'))
    # t = TF.to_tensor(img)
    tf = get_transform()
    geo = GeoposeDataset(transform=tf)
    fn = 'flickr_sge_4019597949_63e5f8ebaa_o'
    sample = geo[fn]
    # sample = {'image': img, 'depth': np.zeros((img.shape[0], img.shape[1]))}
    # sample = tf(sample)
    if is_ken:
        img = sample['og_image'].cuda()
        res = net(img.unsqueeze(0))
        sample = {**sample, 'pred': res}
    else:
        img = sample['image'].cuda()
        res = net(img.unsqueeze(0))
        sample = {**sample, 'pred': res}
        sample = rescale(sample)
    plt.imsave('hills_ken.png', res.squeeze().cpu().numpy(), )
    viz.tensor_imshow(res.squeeze(0))
    plt.colorbar()
    plt.show()
    # infer()
