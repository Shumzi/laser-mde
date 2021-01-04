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


cfg = ConfigHandler()
current_time = datetime.now()
if not cfg['model']['randomize_seed']:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


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


def get_dev():
    """
    Returns: device to be used for torch (preferably GPU).

    """
    if torch.cuda.is_available() and cfg['model']['use_cuda']:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_folder_name():
    """

    Returns: folder name in which to save tensorboard run and model checkpoints.

    """
    cfg_model = cfg['model']
    use_saved = cfg_model['use_saved']
    if use_saved:
        folder_name = cfg_model['path']
        if folder_name.endswith('.pt'):
            return '/'.join(folder_name.split('/')[:-1])
        else:
            return folder_name
    run_name = cfg['train']['run_name']
    cur_time = current_time.strftime("%m_%d_%H-%M-%S")
    return cur_time + '_' + run_name
