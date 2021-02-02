import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
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


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


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


def triple_loss(pred, gt_depth):
    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = Sobel().cuda()
    ones = torch.ones(gt_depth.size(0), 1, gt_depth.size(2), gt_depth.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)
    gt_grad = get_gradient(gt_depth)
    pred_grad = get_gradient(pred)
    gt_grad_dx = gt_grad[:, 0, :, :].contiguous().view_as(gt_depth)
    gt_grad_dy = gt_grad[:, 1, :, :].contiguous().view_as(gt_depth)
    pred_grad_dx = pred_grad[:, 0, :, :].contiguous().view_as(gt_depth)
    pred_grad_dy = pred_grad[:, 1, :, :].contiguous().view_as(gt_depth)
    depth_normal = torch.cat((-gt_grad_dx, -gt_grad_dy, ones), 1)
    output_normal = torch.cat((-pred_grad_dx, -pred_grad_dy, ones), 1)
    loss_depth = torch.log(torch.abs(pred - gt_depth) + 0.5).mean()
    loss_dx = torch.log(torch.abs(pred_grad_dx - gt_grad_dx) + 1).mean()
    loss_dy = torch.log(torch.abs(pred_grad_dy - gt_grad_dy) + 1).mean()
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
    return loss_depth + loss_normal + (loss_dx + loss_dy)


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
