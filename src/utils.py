import os
from os import system, name
from pathlib import Path
import torch
import yaml


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

