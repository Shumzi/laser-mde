# local imports.
from utils import ROOT_DIR
from data_loader import FarsightDataset, ToTensor
import visualize as viz
import model
from utils import DATA_DIR, get_depth_dir, get_img_dir, get_dev
from other_models.tiny_unet import UNet

import sys
import os
import logging
from IPython.display import clear_output

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from trains import Task
from tqdm import tqdm

sys.path.append(os.path.join('../src'))


# task = Task.init(project_name='mde', task_name='test loop')
# logger = task.get_logger()

def train(epochs=2,
          verbose=False,
          batch_size=2,
          val_percent=0.25):
    """
    main training loop.
    """
    print('started')
    writer = SummaryWriter()
    # create dataset
    ds = FarsightDataset(img_dir=get_img_dir(),
                         depth_dir=get_depth_dir(),
                         transform=ToTensor())
    minids = Subset(ds, range(40))
    n_val = int(len(minids) * val_percent)
    n_train = len(minids) - n_val
    train, val = random_split(minids,
                              [n_train, n_val],
                              generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=0)
    val_loader = DataLoader(val,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=0)
    # TODO: fix weird float32 requirement in conv2d to work with uint8. Quantization?
    net = UNet()
    net.to(device=get_dev())
    print('using ', get_dev())
    #     net.apply(weight_init)
    num_batches = len(train_loader)
    print('num_batches: ', num_batches)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    # main training loop.
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [input images, depth maps]
                imgs, gt_depths = data['image'], data['depth']
                optimizer.zero_grad()
                # print('input shape {}, type: {}'.format(inputs.size(), inputs.dtype))
                pred_depth = net(imgs)
                #             print('out shape: {}, gt shape: {}'.format(outputs.size(), gt_depths.size()))
                loss = criterion(pred_depth, gt_depths)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                pbar.update(imgs.shape[0])
                # val statistics. check stats
                if i == num_batches - 1:  # last batch
                    writer.add_scalar('Loss/train', running_loss / num_batches, epoch + 1)
                    #                     clear_output()
                    #                     print('[epoch {}] loss: {:.5f}'.format(epoch + 1, running_loss / num_batches))
                    # val scores
                    val_score = model.eval_net(net, val_loader, criterion)
                    writer.add_scalar('Metric/test', val_score, epoch)
                    logging.info('Validation score (metric?): {}'.format(val_score))

                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)

                    writer.add_images('images', imgs, epoch)
                    writer.add_images('masks/true', gt_depths.unsqueeze(1), epoch)
                    writer.add_images('masks/pred', pred_depth.unsqueeze(1), epoch)
            if verbose or epoch == epochs - 1:
                #             viz.show_batch({**data, 'pred': outputs.detach()})
                viz.show_batch({**data, 'pred': pred_depth.detach()})
                plt.title('epoch {}'.format(epoch))
                plt.show()
    print('Finished Training')
    writer.close()

if __name__=='__main':
    train(10)