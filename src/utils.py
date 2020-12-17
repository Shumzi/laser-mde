import os
from os import system, name
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = \
    Path(r'../data')
USE_CUDA = True


def get_depth_dir():
    """

    Returns: path to depth folder

    """
    return os.path.join(DATA_DIR, 'farsight', 'depth')


def get_img_dir():
    """

    Returns: path to img folder

    """
    return os.path.join(DATA_DIR, 'farsight', 'img')


def get_dev():
    """
    Returns: device to be used for torch (preferably GPU).

    """
    if torch.cuda.is_available() and USE_CUDA:
        return torch.device('cuda')
    else:
        return torch.device('cpu')
