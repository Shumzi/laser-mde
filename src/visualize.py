import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF
from utils import load_checkpoint
import os


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
    To be used when plotting a batch from Dataloader.
    Args:
        batch: dict of batch, each key being a different type of image.
                If dict contains 'name' key, it'll be used as the filenames for each img tuple.
                batch should contain AT LEAST 2 objects to be plotted. we're not playing here.

    Returns: fig of plot.

    Example:
        >> batch = {'image':list_of_images, 
                'depth':list_of_depths,
                'name':list_of_filenames}
        >> show_batch(batch)
        >> plt.show()
    """
    filenames = batch.get('name', None)
    image_type_names, image_types = list(batch.keys()), list(batch.values())  # image_type_names: image, depth, etc.
    batch_size = image_types[0].shape[0]
    tuple_size = len(image_types)  # amount of image types per sample we'll be displaying.
    # e.g. if we have img,depth - then 2.
    u = Unravel(tuple_size, batch_size)
    fig, ax = plt.subplots(tuple_size, batch_size, figsize=(batch_size * 4 + 2, tuple_size * 4))
    # image type names (depth, image, pred, etc.)
    for i, type_name in enumerate(image_type_names):
        ax[u(i * batch_size)].set_ylabel(type_name, fontsize='x-large')
    # titles of filenames
    if filenames is not None:
        for j, fn in enumerate(filenames):
            ax[u(j)].set_title('sample #{}\n{}'.format(j, fn))
    else:
        for j in range(batch_size):
            ax[u(j)].set_title('sample #{}'.format(j))
    # display actual image.
    for i, image_type in enumerate(image_types):
        if torch.is_tensor(image_type):
            image_type = image_type.detach().cpu().numpy()
            if image_type.shape[1] == 4:  # remove alpha/mask
                image_type = image_type[:, :3]
            if image_type.shape[1] == 3:
                # reshape back from C X H X W into H X W X C.
                image_type = image_type.transpose(0, 2, 3, 1)
            elif len(image_type.shape) == 4 and image_type.shape[1] == 1:
                # Grayscale image that wasn't yet squeezed.
                image_type = image_type.squeeze(1)
        if len(image_type.shape) == 2:
            image_type = np.expand_dims(image_type, 0)
        for j, img in enumerate(image_type):
            # TODO: vmin,vmax for pred and gt to be the same.
            im = ax[u(i * batch_size + j)].imshow(img.squeeze())
            if len(img.shape) == 2:  # grayscale
                fig.colorbar(im, ax=ax[u(i * batch_size + j)])
    return fig


def show_sample(sample):
    """
    Same as show_batch, only for a single sample.
    To be used when taking a sample from Dataset.
    Args:
        sample: single sample

    Returns: fig of plot.

    """
    for k, v in sample.items():
        if k == 'name':
            sample[k] = [v]
        else:
            sample[k] = v[np.newaxis]
    return show_batch(sample)


def blend_images(im1, im2):
    """
    blends two PIL images (with .25 alpha for second one).
    Args:
        im1: PIL image.
        im2: PIL image.

    Returns:
        blended pil image.

    """
    # im = Image.fromarray(np.uint8(cm.(im1) * 255))
    if type(im1) == np.ndarray:
        im1 = Image.fromarray(im1)
    elif torch.is_tensor(im1):
        im1 = TF.to_pil_image(im1)
    if type(im2) == np.ndarray:
        im2 = Image.fromarray(im2)
    elif torch.is_tensor(im2):
        im2 = TF.to_pil_image(im2)
    im1 = im1.convert("RGBA")
    im2 = im2.convert("RGBA")
    blended = Image.blend(im1, im2, alpha=.25)
    return blended


def get_sub_batch(batch, subsize):
    """
    return first <subsize> samples in batch.
    Args:
        batch: batch of samples.
        subsize: num of samples to be taken from batch.

    Returns: batch (only smaller)

    """
    minibatch = {}
    for k, v in batch.items():
        minibatch[k] = v[:subsize]
    return minibatch


def tensor_imshow(img):
    """
    plots tensor img on plt. doesn't plt.show it though.
    Args:
        img: Tensor of shape (CxHxW)

    Returns: fig of image.

    """
    return plt.imshow(img.detach().cpu().numpy().transpose((1, 2, 0)))


def show_geopose_sample_with_blend(sample):
    """
    util for showing a specific sample in geopose dataset with blend.
    for debugging purposes.
    Returns: fig of plotted image.

    """
    sample['blend'] = blend_images(sample['image'], sample['depth'])
    return show_sample(sample)


def vis_weight_dist(net, writer, epoch):
    """
    plot weight dist in writer.
    Args:
        net:
        writer:
        epoch:

    Returns: None.

    """
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
        if value.grad is not None:
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
    # writer.add_histogram('values', train_sample['log_pred'].detach().cpu().numpy(), epoch)


def plot_weights():
    """
    plot weights of a net checkpoint in summarywriter (i.e. tensorboard).
    """
    ckpt = load_checkpoint()
    net = ckpt[0]
    writer = SummaryWriter(os.path.join('runs', 'vis_weights'))
    vis_weight_dist(net, writer, 0)
    writer.close()
    print('done')


if __name__ == '__main__':
    pass
    # torch.manual_seed(42)
    # n_samples = 1
    # x = torch.randn(n_samples * 100).view(n_samples, 10, 10)
    # y = torch.randn(n_samples * 100).view(n_samples, 10, 10)
    # show_batch({'x': x, 'y': y})
    # plt.show()
