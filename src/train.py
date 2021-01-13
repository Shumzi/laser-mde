import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from clearml import Task
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
import visualize as viz
from data_loader import FarsightDataset, ToTensor, get_farsight_fold_dataset
from other_models.tiny_unet import UNet
from utils import get_dev, cfg, get_folder_name

logger = logging.getLogger(__name__)
if cfg['misc']['verbose']:
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

if cfg['misc']['use_trains']:
    if cfg['checkpoint']['use_saved']:
        task = Task.get_task(project_name='ariel-mde', task_name=get_folder_name())
    else:
        task = Task.init(project_name='ariel-mde', task_name=get_folder_name())
        config_file = task.connect_configuration(Path('configs.yml'), 'experiment_config')
        cfg = task.connect(cfg)  # enabling configuration override by clearml
    clearml_logger = task.get_logger()


def weight_init(m):
    """
    initialize weights of net to Kaiming and biases to zero,
    since pytorch doesn't do that (assumes leaky relu).

    Usage: net.apply(weight_init)

    Args:
        m: layer from net

    Returns:
        None (initialized weights inplace).

    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
    cfg_checkpoint = cfg['checkpoint']
    cfg_validation = cfg['validation']
    epochs = cfg_train['epochs']
    print_every = cfg_train['print_every']
    save_every = cfg_checkpoint['save_every']
    folder_name = get_folder_name()
    writer = SummaryWriter(os.path.join('runs', folder_name))
    train_loader, val_loader = get_loaders()
    n_batches = len(train_loader)
    # TODO: fix weird float32 requirement in conv2d to work with uint8. Quantization?
    cfg_model = cfg['model']
    cfg_checkpoint = cfg['checkpoint']
    cfg_optim = cfg['optim']
    if cfg_checkpoint['use_saved']:
        net, optimizer, epoch_start, running_loss = load_checkpoint()
        criterion = get_criterion()
        epoch_start = epoch_start + 1  # since we stopped at the last epoch, continue from the next.
    else:
        criterion, net, optimizer = get_net()
        running_loss = 0.0
        epoch_start = 0
    if cfg_optim['use_lr_scheduler']:
        old_lr = optimizer.param_groups[0]['lr']
        scheduler = ReduceLROnPlateau(optimizer, mode='min')
    logger.info('got all params, starting train loop')
    for epoch in range(epoch_start, epochs):  # loop over the dataset multiple times
        net.train()
        with tqdm(total=n_batches, desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for data in train_loader:
                # get the inputs; data is a list of [input images, depth maps]
                img, gt_depth = data['image'], data['depth']
                loss, pred_depth = step(criterion, img, gt_depth, net, optimizer)
                loss_value = loss.item()
                pbar.set_postfix(**{'loss (batch)': loss_value})
                running_loss += loss_value
                pbar.update()

            if cfg_optim['use_lr_scheduler']:
                val_score, val_sample = eval_net(net, val_loader, criterion)
                scheduler.step(val_score)  # possibly plateau LR.
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(fr'old lr: {old_lr}, new lr: {new_lr}')
                old_lr = new_lr
            if epoch % print_every == print_every - 1:
                #     # TODO: maybe add train_val
                if not cfg_optim['use_lr_scheduler']:
                    if cfg_validation['val_round']:
                        val_score, val_sample = eval_net(net, val_loader, criterion)
                    else:
                        val_score = None
                        val_sample = None
                train_loss = running_loss / (print_every * n_batches)
                train_sample = {**data, 'pred': pred_depth}
                print_stats(train_sample, val_sample,
                            train_loss, val_score, epoch, writer)
                running_loss = 0.0
            if save_every is not None and (epoch % save_every == save_every - 1):
                save_checkpoint(epoch, net, optimizer, running_loss)
    print('Finished Training')
    writer.close()
    # TODO: graceful death - checkpoint when exiting run as well.
    if save_every is not None:
        save_checkpoint(epochs - 1, net, optimizer, 0)


def step(criterion, img, gt_depth, net, optimizer):
    """
    single forward and backward step with a specific image batch.

    Returns: loss: float, predicted depth map: torch.Tensor.

    """
    optimizer.zero_grad()
    pred_depth = net(img)
    loss = criterion(pred_depth, gt_depth)
    loss.backward()
    optimizer.step()
    return loss, pred_depth


def eval_net(net, loader, metric):
    """
    Validation stage in the training loop.

    Args:
        net: network being trained
        loader: data loader of validation data
        metric: metric to test validation upon.
    Returns: score of eval based on criterion.

    """
    net.eval()
    n_val = len(loader)
    score = 0
    val_sample = {}
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, gt_depths = batch['image'], batch['depth']
            with torch.no_grad():
                pred_depths = net(imgs)
            score += metric(pred_depths, gt_depths)
            if i == n_val - 1:
                val_sample.update({**batch, 'pred': pred_depths})
                # fig = viz.show_batch({**batch, 'pred': pred_depths})
                # writer.add_images('val/pred', pred_depths.unsqueeze(1), step)
                # writer.add_images('val/gt', gt_depths.unsqueeze(1), step)
            pbar.update()
    score /= n_val
    net.train()
    return score, val_sample


def print_stats(train_sample, val_sample,
                train_loss, val_score, epoch, writer, net=None):
    """
    log statistics and figures of the current log round
    (to be used every 'print_every' in configs)
    Args:
        train_sample: batch sample from train, will be visualized in log.
        val_sample: batch  sample from val, will be visualized in log.
        train_loss: float, loss from train
            (possibly will have separate validation round from current stage)
        val_score: float, score of validation round
        epoch: int, epoch no. in training.
        writer: summaryWriter object, to which we'll output all our figs, scalars, etc.
        net: network being trained, only needed if you want to print hist of weights, etc.

    Returns: None (info is put into Tensorboard file).

    """
    if val_score is not None:
        writer.add_scalars('Loss', {'train': train_loss,
                                    'val': val_score},
                           epoch)
        logger.warning(f'\nValidation loss: {val_score}')
    else:
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
    logger.warning(f'\ntrain loss: {train_loss}')

    print_hist = cfg['validation']['hist']
    if print_hist:
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
        writer.add_histogram('values', train_sample['pred'].detach().cpu().numpy(), epoch)
    logger.info('logging images...')
    fig = viz.show_batch(train_sample)
    fig.suptitle(f'train, epoch {epoch}', fontsize='xx-large')
    # plt.show()
    writer.add_figure(tag='viz/train', figure=fig, global_step=epoch)
    if val_sample is not None:
        fig = viz.show_batch(val_sample)
        fig.suptitle(f'val, epoch {epoch}', fontsize='xx-large')
        # plt.show()
        writer.add_figure(tag='viz/val', figure=fig, global_step=epoch)


def get_net():
    """
    get objects for training the network,
    as specified in configs.yml 'model'

    Returns: (criterion, net, optimizer) where:
        criterion: loss function for optimization.
        net: the network being used for training.
        optimizer:  optimization object (nn.optim)
    """
    cfg_model = cfg['model']
    cfg_checkpoint = cfg['checkpoint']
    cfg_optim = cfg['optim']
    model_name = cfg_model['name'].lower()
    if model_name == 'unet':
        net = UNet()
    elif model_name == 'toynet':
        net = model.toyNet()
    else:
        raise ValueError("can only use UNET or toynet.")
    if cfg_model['weight_init'] and not cfg_checkpoint['use_saved']:
        net.apply(weight_init)
    net.to(device=get_dev())
    print('using ', get_dev())
    # TODO: use loss in configs for loss.
    criterion = get_criterion()
    optimizer = optim.Adam(net.parameters(), lr=cfg_optim['lr'])
    return criterion, net, optimizer


def get_criterion():
    cfg_optim = cfg['optim']
    loss_func_name = cfg_optim['loss'].lower()
    if loss_func_name.startswith('rmsle'):
        logger.info('using rmsle')
        criterion = model.RMSLELoss()
    elif loss_func_name.startswith('mse'):
        criterion = nn.MSELoss()
    else:
        raise ValueError("can only use rmsle or mse")
    return criterion


def get_loaders():
    """
    get data loaders for train set and val set

    Returns: train_loader and val_loader (pytorch dataloaders).

    """
    cfg_train = cfg['train']
    cfg_validation = cfg['validation']
    batch_size = cfg_train['batch_size']
    batch_size_val = cfg['validation']['batch_size']
    val_percent = cfg_validation['val_percent']
    subset_size = cfg_train['subset_size']
    if cfg_train['use_folds']:
        train_split, val_split = get_farsight_fold_dataset(1)
        if subset_size is not None:
            train_size = int(subset_size * (1 - val_percent))
            val_size = int(subset_size * val_percent)
            train_split = Subset(train_split, range(train_size))
            val_split = Subset(val_split, range(val_size))
    else:
        # TODO: generalize dataset to any dataset (hills for example).
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


def save_checkpoint(epoch, net, optimizer, running_loss):
    """
    save a checkpoint of the network for future use.
    location is defined in configs.yml file.
    Args:
        epoch: int.
        net: network object (both weights and model object base is saved).
        optimizer
    # TODO: check rnd. gen is consistent.: optimizer object (only weights are saved).
        running_loss: float, current loss (for possibly future use).

    Returns: None (checkpoint saved).

    """
    logging.info(f'\nsaving checkpoint at epoch {epoch}...')
    folder = get_folder_name()
    folder = os.path.join('../models', folder)
    filename = 'epoch_' + str(epoch).zfill(4) + '.pt'
    if not os.path.exists(folder):
        os.mkdir(folder)
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        logger.warning(f'not saving {full_path} as it already exists.')
        return
    torch.save({
        'model': net,
        'optimizer': optimizer,
        'epoch': epoch,
        'loss': running_loss
    }, full_path)


def load_checkpoint():
    """
    load a saved checkpoint from the training process,
    be it for continued training or inference.
    location from which to load checkpoint is defined in configs.yml file.

    Args:
    Returns: (net, optim, epoch, loss) where:
        net: weighted net
        optim: weighted optimizer
        epoch: updated epoch from checkpoint
        loss: current loss in checkpoint.

    """
    path = os.path.join('..', 'models', cfg['checkpoint']['saved_path'])
    if not path.endswith('.pt'):
        # default to last trained model.
        path = os.path.join(path, max(os.listdir(path)))
    if not os.path.exists(path):
        raise FileNotFoundError
    logging.info(f'loading from {path}')
    checkpoint = torch.load(path)
    net = checkpoint['model']
    optim = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return net, optim, epoch, loss


if __name__ == '__main__':
    """
    lr
    batch_size
    use_bn
    use_double_bn
    dropout
    random_seed
    flip_p
    """
    lr = []
    # UniformParameterRange('config/data_augmentation/horizontal_flip', min_value=0, max_value=1),
    # UniformParameterRange('config/data_augmentation/color_jitter', min_value=0, max_value=1),
    # UniformParameterRange('config/data_augmentation/gaussian_blur', min_value=0, max_value=1),
    # UniformParameterRange('config/data_augmentation/gaussian_noise', min_value=0, max_value=1)
    train()
