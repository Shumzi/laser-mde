from matplotlib import pyplot as plt
from matplotlib import image
import torch
from mpl_toolkits.axes_grid1 import ImageGrid


def show_batch(batch): #TODO: check that you can just do this and then everything is easier.
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
        >> plt.show()
    """
    try:
        filenames = batch.pop('name')
    except:
        filenames = None
    types, images_batches = list(batch.keys()), list(batch.values())  # images_batches: list(batches)
    batch_size = len(images_batches[0])
    type_size = len(images_batches)  # amount of types of images we'll be displaying.
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