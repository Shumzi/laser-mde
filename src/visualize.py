from matplotlib import pyplot as plt
from matplotlib import image
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image


class Unravel:
    def __init__(self, tuple_size, batch_size):
        """
        helper class for unraveling indexes of plots.
        created for edge case of an axes being 1d.
        Args:
            tuple_size: int, no. of images per tuples that will be displayed
            batch_size: int, no. of image tuples in batch.
        """
        self.tuple_size = tuple_size
        self.batch_size = batch_size

    def __call__(self, idx):
        if self.tuple_size == 1 or self.batch_size == 1:
            pos = idx
        else:
            pos = np.unravel_index(idx, (self.tuple_size, self.batch_size))
        return pos


def show_batch(batch):
    """
    plot a batch of samples. can be images + depth + pred, or whatever.
    Args:
        batch: dict of batch, each key being a different type of image.
                If dict contains 'name' key, it'll be used as the filenames for each img tuple.
                batch should contain AT LEAST 2 objects to be plotted. we're not playing here.

    Returns: fig of plot.

    Examples:
        >> batch = {'image':list_of_images, 
                'depth':list_of_depths,
                'name':list_of_filenames}
        >> show_batch(batch)
        >> plt.show()
    """
    try:
        filenames = batch.pop('name')
    except:
        filenames = None
    keys, values = list(batch.keys()), list(batch.values())  # images_batches: list(batches)
    batch_size = len(values[0])
    tuple_size = len(values)  # amount of images in per sample we'll be displaying.
    u = Unravel(tuple_size, batch_size)
    fig, ax = plt.subplots(tuple_size, batch_size, figsize=(batch_size * 4 + 2, tuple_size * 4))
    # image type
    for i, type_name in enumerate(keys):
        ax[u(i * batch_size)].set_ylabel(type_name, fontsize='x-large')
    # sample number
    if filenames is not None:
        for j, fn in enumerate(filenames):
            ax[u(j)].set_title('sample #{}\n{}'.format(j, fn))
        # push back into dict.
        batch['name'] = filenames
    else:
        for j in range(batch_size):
            ax[u(j)].set_title('sample #{}'.format(j))
    # display actual image.
    for i, img_batch in enumerate(values):
        if torch.is_tensor(img_batch):
            img_batch = img_batch.detach().cpu().numpy()
            if img_batch.shape[1] == 3:
                # reshape back from C X H X W into H X W X C.
                img_batch = img_batch.transpose(0, 2, 3, 1)
            elif len(img_batch.shape) == 4 and img_batch.shape[1] == 1:
                # Grayscale image that wasn't yet squeezed.
                img_batch = img_batch.squeeze(1)
        for j, img in enumerate(img_batch):
            ax[u(i * batch_size + j)].imshow(img)
    return fig


def blend_images(im1, im2):
    im1 = im1.convert("RGBA")
    im2 = im2.convert("RGBA")
    blended = Image.blend(im1, im2, alpha=.5)
    return blended


def tensor_imshow(img):
    plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(42)
    n_samples = 1
    x = torch.randn(n_samples * 100).view(n_samples, 10, 10)
    y = torch.randn(n_samples * 100).view(n_samples, 10, 10)
    show_batch({'x': x, 'y': y})
    plt.show()
