import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
# from trains import Task
from tqdm import tqdm

import model
import visualize as viz
from data_loader import FarsightDataset, ToTensor
from other_models.tiny_unet import UNet
from utils import get_depth_dir, get_img_dir, get_dev, cfg
import cProfile

logger = logging.getLogger(__name__)
if cfg['verbose']:
    logger.setLevel(logging.INFO)

# task = Task.init(project_name='mde', task_name='test loop')
# logger = task.get_logger()
def weight_init(m):
    """
    initialize weights of net to Kaiming and biases to zero,
    since pytorch doesn't do that.

    Usage: net.apply(weight_init)

    Args:
        m: layer from net

    Returns:
        None (initialized weights inplace).

    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data   
        print(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.zeros_(m.bias)


def train():
    """
    main train loop. all configurations are taken from the configs.yml file.
    Returns:
        trained net
    """
    logger.info('getting params, dataloaders, etc...')
    cfg_train = cfg['train']
    epochs = cfg_train['epochs']
    run_name = cfg_train['run_name']
    print_every = cfg_train['print_every']
    writer = SummaryWriter(comment=run_name)

    train_loader, val_loader = get_loaders()
    n_batches = len(train_loader)
    # TODO: fix weird float32 requirement in conv2d to work with uint8. Quantization?
    criterion, net, optimizer = get_net()

    logger.info('got all params, starting train loop')
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        with tqdm(total=n_batches, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            running_loss = 0.0
            for data in train_loader:
                # get the inputs; data is a list of [input images, depth maps]
                img, gt_depth = data['image'], data['depth']
                loss, pred_depth = step(criterion, img, gt_depth, net, optimizer)
                loss_val = loss.item()
                logger.info(f'loss: {loss_val}')
                pbar.set_postfix(**{'loss (batch)': loss_val})
                running_loss += loss_val
                pbar.update()

            if epoch % print_every == print_every - 1:
                if cfg_train['val_round']:
                    val_score = model.eval_net(net, val_loader, criterion, writer, epoch)
                    # TODO: maybe add train_val
                else:
                    val_score = None
                print_stats(net, data, epoch, val_score,
                            pred_depth, running_loss, writer) 
                running_loss = 0.0
    print('Finished Training')
    writer.close()
    return gt_depth, pred_depth


def step(criterion, img, gt_depth, net, optimizer):
    optimizer.zero_grad()
    pred_depth = net(img)
    loss = criterion(pred_depth, gt_depth)
    loss.backward()
    optimizer.step()
    return loss, pred_depth


def print_stats(net, data, epoch, val_score,
                pred_depth, running_loss, writer):

    print_every = cfg['train']['print_every']
    writer.add_scalar('Loss/train', running_loss / print_every, epoch + 1)
    logger.warning('\ntrain loss: {}'.format(running_loss / print_every))

    if val_score is not None:
        writer.add_scalar('Metric/test', val_score, epoch)
        logger.warning('\nValidation loss (metric?): {}'.format(val_score))

    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)

    writer.add_histogram('values', pred_depth.detach().cpu().numpy(), epoch)
    fig = viz.show_batch({**data, 'pred': pred_depth.detach()})
    fig.suptitle(f'step {epoch}', fontsize='xx-large')
    writer.add_figure(tag='epoch/end', figure=fig, epoch=epoch)
    writer.add_images('images', data['image'], epoch)
    writer.add_images('masks/gt', data['depth'].unsqueeze(1), epoch)
    writer.add_images('masks/pred', pred_depth.unsqueeze(1), epoch)


def get_net():
    cfg_model = cfg['model']
    net = UNet()
    net.to(device=get_dev())
    print('using ', get_dev())
    if cfg_model['weight_init']:
        net.apply(weight_init)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg_model['lr'])
    return criterion, net, optimizer


def get_loaders():
    cfg_train = cfg['train']
    batch_size = cfg_train['batch_size']
    val_percent = cfg_train['val_percent']
    subset_size = cfg_train['subset_size']
    ds = FarsightDataset(transform=ToTensor())
    if subset_size is not None:
        ds = Subset(ds, range(subset_size))
    n_val = int(len(ds) * val_percent)
    n_train = len(ds) - n_val
    train_split, val_split = random_split(ds,
                                          [n_train, n_val],
                                          generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_split,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=0)
    val_loader = DataLoader(val_split,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=0)
    return train_loader, val_loader


if __name__ == '__main__':
    train()
    # cProfile.run('train()', sort='tottime')
