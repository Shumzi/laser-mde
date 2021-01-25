from train import get_geopose_loaders
from tqdm import tqdm
import numpy as np
from utils import cfg


def get_mean_std():
    train_loader, _ = get_geopose_loaders()
    means = {'image': 0, 'depth': 0}
    stds = {'image': 0, 'depth': 0}
    with tqdm(total=(3111 * 0.8 / cfg['train']['batch_size']), unit='batch') as pbar:
        for sample in train_loader:
            depth = sample['depth']
            image = sample['image']
            batch_samples = image.size(0)  # batch size (the last batch can have smaller size!)
            depth = depth.view(batch_samples, depth.size(1), -1)
            image = image.view(batch_samples, image.size(1), -1)
            means['depth'] += np.nanmean(depth, 2).sum(0)  # some depths are bad... have some random nan values.
            stds['depth'] += np.nanstd(depth, 2).sum(0)
            means['image'] += image.mean(2).sum(0)
            stds['image'] += image.std(2).sum(0)
            pbar.set_postfix(**{'mean': means['image'], 'std': stds['image']})
            pbar.update()
    means['depth'] /= len(train_loader.dataset)
    stds['depth'] /= len(train_loader.dataset)
    means['image'] /= len(train_loader.dataset)
    stds['image'] /= len(train_loader.dataset)
    print(means, stds)
    return means, stds


def save_mean_std():
    means, stds = get_mean_std()
    # TODO: FIX SHITTY ROTATIONS MOTHERFUCKER LIKE
    #  flickr_sge_4827030585_76d2db185e_4142_30342452@N07_grid_1_0.004_0.004.xml_0_1_0.506104
    #  for example
    print(means['image'], stds['image'])
    with open('mean_var.npy', 'wb') as f:
        np.save(f, means)
        np.save(f, stds)


# def read_mean_std:
save_mean_std()
# import numpy as np
# from data_loader import GeoposeDataset
# from prepare_data import ResizeToImgShape
# import visualize as viz
# ds = GeoposeDataset(transform=ResizeToImgShape())
# #
# sample = ds['flickr_sge_3620140534_5c19136dd3_o']
# sample['blend'] = np.array(viz.blend_images(sample['image'], sample['depth']))
# viz.show_sample(sample)
# plt.show()
