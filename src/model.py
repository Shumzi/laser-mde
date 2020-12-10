import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# making unet with 1channel output. eigen_net uses way more parameters,
# don't think my gpu can handle it.


class UNetConvBlock(nn.Module):
    """
    this block does the double conv in the unet paper.
    downsampling and upsampling are done outside this module.
    """

    def __init__(self, in_channel, out_channel):
        super(UNetConvBlock, self).__init__()

    # def forward(self, x):


# class ToyNet(nn.Module):
#     def __init__(self):
#         super(ToyNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         return x


class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)
    
def toyNet():
    return nn.Sequential(
#         WeightValues('start'),
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
#         WeightValues('conv1'),
        nn.Conv2d(6, 4, 3),
        nn.ReLU(),
#         WeightValues('conv2'),
        nn.ConvTranspose2d(4, 6, 3),
        nn.ReLU(),
#         WeightValues('convT1'),
        nn.ConvTranspose2d(6, 1, 5),
        nn.ReLU(),
#         WeightValues('convT2'),
        Squeeze()
    )

class WeightValues(nn.Module):
    """
    helper class to see mean and std at some layer.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name
    def forward(self, x):
        print('weights at {}: mean: {}, std: {}'.format(self.name,x.mean(),x.std()))
        return x
    
def eval_net(net, loader, criterion, summary_writer):
    """
    Validation stage in the training loop.

    Args:
        net: network being validated
        loader:
        criterion:
        summary_writer:

    Returns:

    """
    net.eval()
    n_val = len(loader)
    score = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, gt_depths = batch['image'], batch['depth']
            with torch.no_grad():
                pred_depths = net(imgs)
                # writer.
            score += criterion(pred_depths, gt_depths)
            pbar.update()
    score /= n_val
    net.train()
    return score