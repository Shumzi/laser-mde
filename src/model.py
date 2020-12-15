import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tqdm import tqdm
import visualize as viz
from torch.utils.tensorboard import SummaryWriter

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
    
    
def eval_net(net, loader, metric):
    """
    Validation stage in the training loop.

    Args:
        net: network being validated
        loader: data loader of validation data
        metric: metric to test validation upon.

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
                # writer.
            score += metric(pred_depths, gt_depths)
            pbar.update()
    score /= n_val
#     viz.
    net.train()
    return score