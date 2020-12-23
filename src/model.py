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


def eval_net(net, loader, metric, writer, step):
    """
    Validation stage in the training loop.

    Args:
        net: network being trained
        loader: data loader of validation data
        metric: metric to test validation upon.
        writer: SummaryWriter for tensorboard.
        step: int, global step no. for tensorboard.
    Returns: score of eval based on criterion.

    """
    net.eval()
    n_val = len(loader)
    score = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, gt_depths = batch['image'], batch['depth']
            with torch.no_grad():
                pred_depths = net(imgs)
            score += metric(pred_depths, gt_depths)
            if i == n_val - 1:
                # fig = viz.show_batch({**batch, 'pred': pred_depths})
                writer.add_images('val/pred', pred_depths.unsqueeze(1), step)
                writer.add_images('val/gt', gt_depths.unsqueeze(1), step)
            pbar.update()
    score /= n_val
    net.train()
    return score


class RMSLELoss(nn.Module):
    '''
    root mean square log error.
    Currently explodes when using small values between 0 and 1.
    '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        # plus 1 since log(0) is undefined.
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class EigenDepthLoss(nn.Module):
    '''
    eigen depth which promotes structural consistency.
    '''
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, gt):
        d = pred - gt
        pass


if __name__ == '__main__':
    loss = RMSLELoss()
    print(loss(torch.tensor(math.exp(1), dtype=torch.float32), torch.tensor(1, dtype=torch.float32)))
