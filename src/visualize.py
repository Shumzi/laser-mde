from matplotlib import pyplot as plt
from matplotlib import image
import torch
from mpl_toolkits.axes_grid1 import ImageGrid

def plt_image_pair(img_path, depth_path):
    """
    plot single image pair (image+depth) to tensorboard
    """
    pass


def plt_batch(img_batch, depth_batch):
    """
    plot batch of image and depth pairs to tensorboard
    """
    pass


def plt_prediction(img, pred_depth, ground_truth):
    """
    plot image + prediction vs ground truth to tboard.
    """
    pass


# Helper function to show a batch
def show_depths_batch(sample_batched):
    """Show image with depths for a batch of samples."""
    # TODO: deal with batch_size=1
    images_batch, depths_batch, names = \
        sample_batched['image'], sample_batched['depth'], sample_batched['name']
    batch_size = len(images_batch)
    print('batch size: ', batch_size)
    fig, ax = plt.subplots(2, batch_size, figsize=(batch_size * 4, 8))
    # plt.tight_layout()
    ax[0, 0].set_ylabel('image', fontsize='x-large')
    ax[1, 0].set_ylabel('depth', fontsize='x-large')
    for i, img in enumerate(images_batch):
        ax[0, i].set_title('sample #{}\n{}'.format(i, names[i]))
        ax[0, i].imshow(img.numpy().transpose((1, 2, 0)))
    for i, depth in enumerate(depths_batch):
        ax[1, i].imshow(depth)
    print('depth min/max values: ', torch.min(depths_batch), torch.max(depths_batch))

