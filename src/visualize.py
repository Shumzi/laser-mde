from matplotlib import pyplot as plt
from matplotlib import image
import torch
from mpl_toolkits.axes_grid1 import ImageGrid


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


def show_image_triplet(data, output):
    """same as show depths, only contains prediction image as well."""
    show_depths_batch(data)
    batch_size = output.shape[0]
    fig, ax = plt.subplots(1, batch_size, figsize=(batch_size * 4, 4))
    ax[0].set_ylabel('prediction', fontsize='x-large')
    for i, depth in enumerate(output):
        ax[i].imshow(depth)


def show_batch(**batch): #TODO: check that you can just do this and then everything is easier.
    """
    plot a batch of samples. can be images + depth + pred, or whatever.
    input:
    Args:
        batch: dict of batch, each key being a different type of image.
                If dict contains 'name' key, it'll be used as the filenames for each img tuple.

    Returns: fig of plot.

    Examples:
        >> batch = {'image':list_of_images, 
                'depth':list_of_depths,
                'name':list_of_filenames}
        >> show_batch(batch)
    """
    try:
        filenames = batch.pop('name')
    except:
        filenames = None
    types, images_batches = list(batch.keys()), list(batch.values())  # images_batches: list(batches)
    batch_size = len(images_batches[0])
    type_size = len(images_batches)  # amount of types of images we'll be displaying.
    print('batch size: ', batch_size)
    fig, ax = plt.subplots(type_size, batch_size, figsize=(batch_size * 4, type_size * 4))
    # plt.tight_layout()
    # image type
    for i, type_name in enumerate(types):
        ax[i, 0].set_ylabel(type_name, fontsize='x-large')
    # sample number

    if filenames is not None:
        for j, fn in enumerate(filenames):
            ax[0, j].set_title('sample #{}\n{}'.format(j, fn))
        # push back into dict.
        batch['name'] = filenames
    else:
        for j in range(batch_size):
            ax[0, j].set_title('sample #{}'.format(j))
    # display actual image.
    for i, img_batch in enumerate(images_batches):
        if torch.is_tensor(img_batch):
            img_batch = img_batch.detach().cpu().numpy()
            if img_batch.shape[1] == 3:
                # reshape back from C X H X W into H X W X C.
                img_batch = img_batch.transpose(0, 2, 3, 1)
        for j, img in enumerate(img_batch):
            ax[i, j].imshow(img)
    return fig