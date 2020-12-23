import logging
import os
import sys
from datetime import datetime

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
from utils import get_depth_dir, get_img_dir, get_dev, cfg, current_time
import cProfile

logger = logging.getLogger(__name__)
if cfg['verbose']:
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)


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

    print_every = cfg_train['print_every']
    save_every = cfg_train['save_every']
    folder_name = get_folder_name()
    writer = SummaryWriter(os.path.join('runs', folder_name))

    train_loader, val_loader = get_loaders()
    n_batches = len(train_loader)
    # TODO: fix weird float32 requirement in conv2d to work with uint8. Quantization?
    criterion, net, optimizer = get_net()
    if cfg['model']['use_saved']:
        net, optimizer, epoch_start, running_loss = load_saved_model(net, optimizer)
        epoch_start = epoch_start + 1  # since we stopped at the last epoch, continue from the next.
    else:
        epoch_start = 0
    running_loss = 0.0
    logger.info('got all params, starting train loop')
    for epoch in range(epoch_start, epochs):  # loop over the dataset multiple times
        net.train()
        with tqdm(total=n_batches, desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for data in train_loader:
                # get the inputs; data is a list of [input images, depth maps]
                img, gt_depth = data['image'], data['depth']
                loss, pred_depth = step(criterion, img, gt_depth, net, optimizer)
                loss_val = loss.item()
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
                            pred_depth, running_loss, n_batches, writer)
                running_loss = 0.0
            if save_every is not None and (epoch % save_every == save_every - 1):
                if cfg_train['save']:
                    save_checkpoint(epoch, net, optimizer, running_loss)
    print('Finished Training')
    writer.close()
    # TODO: graceful death - checkpoint when exiting run as well.
    if cfg_train['save']:
        save_checkpoint(epochs, net, optimizer, 0)


def get_folder_name():
    """

    Returns: folder name in which to save tensorboard run and model checkpoints.

    """
    cfg_fn = cfg['train']['foldername']
    if cfg_fn is not None:
        return cfg_fn
    run_name = cfg['train']['run_name']
    cur_time = current_time.strftime("%m_%d_%H-%M-%S")
    return cur_time + '_' + run_name


def save_checkpoint(epoch, net, optimizer, running_loss):
    logging.info(f'saving checkpoint at epoch {epoch}...')
    folder = get_folder_name()
    run_name = cfg['train']['run_name']
    folder = os.path.join('../models', folder)
    filename = 'epoch' + str(epoch) + '.pt'
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss
    }, os.path.join(folder, filename))


def step(criterion, img, gt_depth, net, optimizer):
    optimizer.zero_grad()
    pred_depth = net(img)
    loss = criterion(pred_depth, gt_depth)
    loss.backward()
    optimizer.step()
    return loss, pred_depth


def print_stats(net, data, epoch, val_score,
                pred_depth, running_loss, n_batches, writer):
    print_every = cfg['train']['print_every']
    writer.add_scalar('Loss/train', running_loss / (print_every * n_batches), epoch + 1)
    logger.warning('\ntrain loss: {}'.format(running_loss / (print_every * n_batches)))

    if val_score is not None:
        writer.add_scalar('Loss/val', val_score, epoch)
        logger.warning(f'\nValidation loss: {val_score}')

    print_hist = cfg['evaluate']['hist']
    if print_hist:
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
        writer.add_histogram('values', pred_depth.detach().cpu().numpy(), epoch)
    logger.info('logging images...')
    fig = viz.show_batch({**data, 'pred': pred_depth.detach()})
    fig.suptitle(f'epoch {epoch}', fontsize='xx-large')
    writer.add_figure(tag='epoch/end', figure=fig, global_step=epoch)
    # writer.add_images('masks/gt', data['depth'].unsqueeze(1), epoch)
    # writer.add_images('masks/pred', pred_depth.unsqueeze(1), epoch)


def get_net():
    """
    get objects for training the network.
    Returns:
        criterion: loss function for optimization.
        net: the network being used for training.
        optimizer:  optimization object (nn.optim)
    """
    cfg_model = cfg['model']
    model_name = cfg_model['name'].lower()
    if model_name == 'unet':
        net = UNet()
    elif model_name == 'toynet':
        net = model.toyNet()
    if cfg_model['weight_init'] and not cfg_model['use_saved']:
        net.apply(weight_init)
    net.to(device=get_dev())
    print('using ', get_dev())
    # TODO: use loss in configs for loss.
    loss_func_name = cfg_model['loss'].lower()
    if loss_func_name.startswith('rmsle'):
        logger.info('using rmsle')
        criterion = model.RMSLELoss()
    elif loss_func_name.startswith('mse'):
        criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg_model['lr'])
    return criterion, net, optimizer


def get_loaders():
    cfg_train = cfg['train']
    batch_size = cfg_train['batch_size']
    batch_size_val = cfg['evaluate']['batch_size']
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
    # TODO: check rnd. gen is consistent.
    # TODO: make optional to use manual seed or random at some point. (same for DL?)
    train_loader = DataLoader(train_split,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=0)
    val_loader = DataLoader(val_split,
                            shuffle=False,
                            batch_size=batch_size_val,
                            num_workers=0)
    return train_loader, val_loader


def load_saved_model(net, optim):
    path = cfg['model']['path']
    if not path.endswith('.pt'):
        # default to last trained model.
        path = path + os.listdir(path)[-1]
    logging.info(f'loading from {path}')
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return net, optim, epoch, loss


if __name__ == '__main__':
    train()

    # criterion, net, optim = get_net()

    # cProfile.run('train()', sort='tottime')
