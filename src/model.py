import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tqdm import tqdm
import visualize as viz
from torch.utils.tensorboard import SummaryWriter
import math


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

class EigenCoarse(nn.Module):
    """
    based on eigen et al. 2014 https://arxiv.org/pdf/1406.2283.pdf
    coarse net to give rough shape of depth,

    """
class WeightValues(nn.Module):
    """
    helper class to see mean and std at some layer.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print('weights at {}: mean: {}, std: {}'.format(self.name, x.mean(), x.std()))
        return x


class RMSLELoss(nn.Module):
    """
    root mean square log error.
    taken from: https://discuss.pytorch.org/t/rmsle-loss-function/67281
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        # plus 1 since log(0) is undefined.
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class EigenDepthLoss(nn.Module):
    """
    eigen depth which promotes structural consistency.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, gt):
        d = pred - gt
        pass


if __name__ == '__main__':
    loss = RMSLELoss()
    print(loss(torch.tensor(math.exp(1), dtype=torch.float32), torch.tensor(1, dtype=torch.float32)))
