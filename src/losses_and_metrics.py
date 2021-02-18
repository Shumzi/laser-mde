import numpy as np
import torch
from torch import nn as nn


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


class RMSLEScaleInvariantError(nn.Module):
    """
    root mean square log error.
    taken from: https://discuss.pytorch.org/t/rmsle-loss-function/67281
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        # plus 1 since log(0) is undefined.
        d_i = torch.log(pred + 1) - torch.log(actual + 1)

        n = torch.prod(torch.tensor(pred.shape))
        return torch.sum(d_i ** 2) / n - (torch.sum(d_i) ** 2) / (n ** 2)


def imgrad(img):
    """
    calculates gradient of image
    from https://github.com/haofengac/MonoDepth-FPN-PyTorch
    Args:
        img:

    Returns: grad_y, grad_x of same shape as img.

    """
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x


def imgrad_yx(img):
    """
    wrapper for imgrad, just gives you all gradients in one big block.
    Args:
        img: img to calc gradients on

    Returns: grads of shape (N,2*C,H*W). 2*C bc of stacking of x and y grads for each channel.

    """
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, pred, gt):
        """
        calculate l1 norm between gradients of pred and gt
        Args:
            gt:
            pred:

        Returns: l1 norm (scalar, float)

        """
        grad_fake = imgrad_yx(pred)
        grad_real = imgrad_yx(gt)
        return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred, gt_depth, num=1):
        """
        compute accuracy of prediction. Accuracy in depth terms is
        % of pixels s.t. max(yhat/y, y/yhat) = delta < 1.25**num
        see https://arxiv.org/pdf/2003.06620.pdf, B.  Evaluation metrics)
        Args:
            num: power of 1.25.

        Returns: (float [0,1]), accuracy.

        """
        disparities = torch.max(pred / gt_depth, gt_depth / pred) < (1.25 ** num)
        return disparities.sum() / disparities.numel()


if __name__ == '__main__':

    loss = RMSLELoss()
    pred = torch.tensor([600.])
    gt = torch.tensor([1000.])
    print(loss(pred, gt))
    del_acc = Accuracy()
    del_acc(pred, gt)
