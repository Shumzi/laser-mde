import gzip
import shutil
import os
from pypfm import PFMLoader
from pathlib import Path
import glob
from matplotlib import pyplot as plt
from os.path import join
import numpy as np


def get_np_from_pfm(fn_gz):
    fn_gz = Path(fn_gz)
    fn_out = fn_gz.with_suffix('.Pf')
    if not os.path.exists(fn_out):
        with gzip.open(fn_gz, 'rb') as f_in:
            with open(fn_out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    loader = PFMLoader()
    return loader.load_pfm(fn_out)


def get_img_and_depth(folder_name):
    """
    returns the original image and it's depth map as a np array.
    return: (img, depth_img: np array)
    """
    folder_name = str(folder_name)
    img = plt.imread(glob.glob(folder_name + '/photo*')[0])
    depth = get_np_from_pfm(folder_name + '/cyl/distance_crop.pfm.gz')[::-1]
    #     plt.savefig(depth)
    return img, depth


if __name__ == '__main__':
    path = '../../data/geoPose3K/eth_ch1_2011-10-04_14_25_54_01024'
    loader = PFMLoader()
    data = loader.load_pfm(join(path, 'distance_crop.pfm'))[::-1]
    plt.imshow(data, cmap=plt.cm.gray)
    plt.imsave(join(path, 'depth.png'), data, cmap="gray")
    # data = plt.imread('../geoPose3K/28561570606/depth.png')
    plt.show()
