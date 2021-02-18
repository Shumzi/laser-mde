import os
import torch
import yaml
from datetime import datetime
import random
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
cfg = yaml.safe_load(open('configs.yml', 'r', encoding='utf-8'))


def set_cfg(new_cfg):
    """
    change cfg file within run. a little shady.
    Args:
        new_cfg:

    Returns: None (cfg is changed for everyone).

    """
    global cfg
    cfg = new_cfg


current_time = datetime.now()
rnd_seed = cfg['misc']['random_seed']
if rnd_seed is not None:
    torch.manual_seed(rnd_seed)
    torch.cuda.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)


def get_depth_dir():
    """

    Returns: path to depth folder

    """
    return os.path.join(cfg['dataset']['data_dir'], cfg['dataset']['depth_foldername'])


def get_img_dir():
    """

    Returns: path to img folder

    """
    return os.path.join(cfg['dataset']['data_dir'], cfg['dataset']['img_foldername'])


def get_test_dir():
    """
        Returns: path to test folder
    """
    return os.path.join(cfg['dataset']['data_dir'], cfg['dataset']['test_foldername'])


def get_dev():
    """
    Returns: device to be used for torch (preferably GPU).

    """
    if torch.cuda.is_available() and cfg['misc']['use_cuda']:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_folder_name():
    """

    Returns: folder name in which to save tensorboard run and model checkpoints.

    """
    cfg_checkpoint = cfg['checkpoint']
    use_saved = cfg_checkpoint['use_saved']
    if use_saved:
        folder_name = cfg_checkpoint['saved_path']
        if folder_name.endswith('.pt'):
            return '/'.join(folder_name.split('/')[:-1])
        else:
            return folder_name
    run_name = cfg_checkpoint['run_name']
    if run_name is None:
        run_name = input('enter run name: ')
        cfg_checkpoint['run_name'] = run_name
    return run_name


def save_checkpoint(epoch, net, optimizer, running_loss):
    """
    save a checkpoint of the network for future use.
    location is defined in configs.yml file.
    Args:
        epoch: int.
        net: network object (both weights and model object base is saved).
        optimizer:
        running_loss:

    Returns: None (checkpoint saved).

    """
    logger.info(f'\nsaving checkpoint at epoch {epoch}...')
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


def load_checkpoint(path=None):
    """
    load a saved checkpoint from the training process,
    be it for continued training or inference.
    location from which to load checkpoint is defined in configs.yml file.

    Args:
        path (string, optional): specify manual location of checkpoint.
                                 defaults to path in configs file.
    Returns: (net, optim, epoch, loss) where:
        net: weighted net
        optim: weighted optimizer
        epoch: updated epoch from checkpoint
        loss: current loss in checkpoint.

    """
    if path is None:
        path = os.path.join('..', 'models', cfg['checkpoint']['saved_path'])
    if not path.endswith('.pt'):
        # default to last trained model.
        path = os.path.join(path, max(os.listdir(path)))
    if not os.path.exists(path):
        raise FileNotFoundError
    logger.info(f'loading from {path}')
    checkpoint = torch.load(path)
    net = checkpoint['model']
    optim = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return net, optim, epoch, loss
