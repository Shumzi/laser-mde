import os
from pathlib import Path
ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = \
    Path(r'/media/ariel/6EEE7AB0EE7A7067/Users/Ariel/Documents/DS הכשרה_/15. MDE with laser reference '
         r'point/laser-mde/data')


def get_depth_dir():
    """return path to depth folder"""
    return os.path.join(DATA_DIR, 'farsight', 'depth')


def get_img_dir():
    """retuns path to img folder"""
    return os.path.join(DATA_DIR, 'farsight', 'img')