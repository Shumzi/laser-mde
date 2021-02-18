import gzip
import shutil
import os
from pypfm import PFMLoader
from pathlib import Path
import glob
from matplotlib import pyplot as plt
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
    '''
    returns the original image and it's depth map as a np array.
    return: (img, depth_img: np array)
    '''
    folder_name = str(folder_name)
    img = plt.imread(glob.glob(folder_name+'/photo*')[0])
    depth = get_np_from_pfm(folder_name+'/cyl/distance_crop.pfm.gz')[::-1]
#     plt.savefig(depth)
    return img, depth