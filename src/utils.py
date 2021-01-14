import os
from os import system, name
from pathlib import Path
import torch
import yaml
from datetime import datetime
import random
import numpy as np


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigHandler(metaclass=Singleton):
    """
    class handler for configs file of project.
    """

    def __init__(self):
        with open('configs.yml', 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except Exception as e:
                print('Error reading the config file')

    def __getitem__(self, attr):
        return self.config[attr]


cfg = yaml.safe_load(open('/home/bina/PycharmProjects/laser-mde/src/configs.yml', 'r'))


def set_cfg(new_cfg):
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
    # cur_time = current_time.strftime("%m_%d_%H-%M-%S")
    # cfg_checkpoint['run_name'] = run_name + cur_time
    return run_name
