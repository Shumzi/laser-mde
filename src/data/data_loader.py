import logging
import numpy as np
import pandas as pd
import os
import sys
import PIL
from utils.definitions import ROOT_DIR
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot

# logging.basicConfig(level=logging.INFO)
data_dir = str(ROOT_DIR) + '/data/farsight/'

def get_train_test():
    """ returns train test split defined by text files in farsight paper
    """
    logger = logging.getLogger(__name__)
    logger.warning('getting train and test data')
    # load image as pixel array
    data = image.imread(os.path.join(data_dir, 'img/houston_1_4_H_-16_000034.png'))
    # summarize shape of the pixel array
    print(data.dtype)
    print(data.shape)
    # display the array of pixels as an image
    pyplot.imshow(data)
    pyplot.show()


if __name__ == '__main__':
    print('Hello')
    get_train_test()