import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from clearml import Task
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
import model
import visualize as viz
from data_loader import FarsightDataset, GeoposeDataset, FarsightToTensor, get_farsight_fold_dataset
from fcrn_pytorch.utils import loss_berhu
from other_models.tiny_unet import UNet
from prepare_data import crop_to_aspect_ratio_and_resize, pad_and_center, reverseMinMaxScale
import prepare_data
from utils import get_dev, cfg, get_folder_name, set_cfg
from torchvision.transforms import Compose
from fcrn_pytorch.fcrn import FCRN
from fcrn_pytorch.weights import load_weights
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)
if cfg['misc']['verbose']:
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)


def weight_init(m):
    """
    initialize weights of net to Kaiming and biases to zero,
    since pytorch doesn't do that (assumes leaky relu).

    Usage: net.apply(weight_init)

    Args:
        m: layer from net

    Returns:
        None (initialized weights inplace).

    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias:
            nn.init.zeros_(m.bias)
        else:
            logger.info(f'no bias for {m}')


def train():
    """
    main train loop. all configurations are taken from the configs.yml file.
    Returns:
        trained net
    """
    logger.info('getting params, dataloaders, etc...')
    cfg_train = cfg['train']
    cfg_checkpoint = cfg['checkpoint']
    cfg_validation = cfg['validation']
    epochs = cfg_train['epochs']
    print_every = cfg_train['print_every']
    save_every = cfg_checkpoint['save_every']
    # use_writer = cfg['misc']['use_writer']
    folder_name = get_folder_name()
    # if use_writer:
    writer = SummaryWriter(os.path.join('runs', folder_name))
    loaders = get_loaders()
    train_loader, val_loader = None, None
    if len(loaders) == 3:
        train_loader, val_loader, depth_postprocessing = loaders
    elif len(loaders) == 2:
        train_loader, val_loader = loaders
        depth_postprocessing = None
    assert train_loader is not None and val_loader is not None, "problem with loader."
    n_batches = len(train_loader)
    # TODO: fix weird float32 requirement in conv2d to work with uint8. Quantization?
    cfg_model = cfg['model']
    cfg_checkpoint = cfg['checkpoint']
    cfg_optim = cfg['optim']
    if cfg_checkpoint['use_saved']:
        net, optimizer, epoch_start, running_loss = load_checkpoint()
        criterion = get_criterion()
        epoch_start = epoch_start + 1  # since we stopped at the last epoch, continue from the next.
    else:
        criterion, net, optimizer = get_net()
        running_loss = 0.0
        epoch_start = 0
    if cfg_optim['use_lr_scheduler']:
        old_lr = optimizer.param_groups[0]['lr']
        scheduler = ReduceLROnPlateau(optimizer, mode='min')
    logger.info('got all params, starting train loop')
    for epoch in range(epoch_start, epochs):  # loop over the dataset multiple times
        net.train()
        with tqdm(total=n_batches, desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for data in train_loader:
                # get the inputs; data is a list of [input images, depth maps]
                img, gt_depth = data['image'], data['depth']
                if cfg['dataset']['use_mask']:
                    assert 'mask' in data, 'no mask but required mask'
                    mask = data['mask']
                else:
                    mask = None
                loss, pred_depth = step(criterion, img, gt_depth, net, optimizer, mask)
                loss_value = loss.item()
                assert loss_value == loss_value, 'loss is nan! tf?'
                pbar.set_postfix(**{'loss (batch)': loss_value})
                running_loss += loss_value
                pbar.update()

            if cfg_optim['use_lr_scheduler']:
                val_score, val_sample = eval_net(net, val_loader, criterion)
                scheduler.step(val_score)  # possibly plateau LR.
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(fr'old lr: {old_lr}, new lr: {new_lr}')
                old_lr = new_lr
            if epoch % print_every == print_every - 1:
                #     # TODO: maybe add train_val
                if not cfg_optim['use_lr_scheduler']:
                    if cfg_validation['val_round']:
                        assert cfg_validation[
                                   'val_percent'] is not None, 'required val_round but didn\'t give a split size'
                        val_score, val_sample = eval_net(net, val_loader, criterion)
                    else:
                        val_score = None
                        val_sample = None
                train_loss = running_loss / (print_every * n_batches)
                # data['log_gt_depth'] = data['depth']
                # del data['depth']
                # del data['image']
                # TODO: see how to save og image for printing w.o doing it for every batch.
                train_sample = {**data, 'pred': pred_depth}
                if cfg['validation']['hist']:
                    viz_net = net
                else:
                    viz_net = None
                if depth_postprocessing:
                    logger.info('post-processing prediction and depth.')
                    train_sample['pred'] = depth_postprocessing(train_sample['pred'])
                    train_sample['depth'] = depth_postprocessing(train_sample['depth'])
                print_stats(train_sample, val_sample,
                            train_loss, val_score, epoch,
                            writer, viz_net)
                running_loss = 0.0
            if save_every is not None and (epoch % save_every == save_every - 1):
                save_checkpoint(epoch, net, optimizer, running_loss)
    print('Finished Training')
    writer.close()
    # TODO: graceful death - checkpoint when exiting run as well.
    if save_every is not None:
        save_checkpoint(epochs - 1, net, optimizer, 0)


def step(criterion, img, gt_depth, net, optimizer, mask=None):
    """
    single forward and backward step with a specific image batch.

    Returns: loss: float, predicted depth map: torch.Tensor.

    """

    optimizer.zero_grad()
    pred = net(img)
    # loss = criterion(pred.squeeze(), gt_depth.squeeze())
    # pred.retain_grad()
    if mask is not None:
        loss = criterion(pred * mask, gt_depth * mask)
    else:
        loss = criterion(pred, gt_depth)
        # pred *= mask.squeeze(1)
        # pred.register_hook(lambda grad: grad * mask)
    #     # TODO: maybe just do criterion on pred and gt * mask?
    # else:
    # torch.autograd.set_detect_anomaly(True)
    loss.backward()
    optimizer.step()
    return loss, pred


def eval_net(net, loader, metric):
    """
    Validation stage in the training loop.

    Args:
        net: network being trained
        loader: data loader of validation data
        metric: metric to test validation upon.
    Returns: score of eval based on criterion.

    """
    net.eval()
    n_val = len(loader)
    score = 0
    val_sample = {}
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, gt_depths = batch['image'], batch['depth']
            with torch.no_grad():
                # .unsqueeze(1)
                pred_depths = net(imgs)
            score += metric(pred_depths, gt_depths)
            if i == n_val - 1:
                val_sample.update({**batch, 'pred': pred_depths})
                # fig = viz.show_batch({**batch, 'pred': pred_depths})
                # writer.add_images('val/pred', pred_depths.unsqueeze(1), step)
                # writer.add_images('val/gt', gt_depths.unsqueeze(1), step)
            pbar.update()
    score /= n_val
    net.train()
    return score, val_sample


def print_stats(train_sample, val_sample,
                train_loss, val_score, epoch,
                writer, net=None):
    """
    log statistics and figures of the current log round
    (to be used every 'print_every' in configs)
    Args:
        train_sample: batch sample from train, will be visualized in log.
        val_sample: batch  sample from val, will be visualized in log.
        train_loss: float, loss from train
            (possibly will have separate validation round from current stage)
        val_score: float, score of validation round
        epoch: int, epoch no. in training.
        writer: summaryWriter object, to which we'll output all our figs, scalars, etc.
        net: network being trained, only needed if you want to print hist of weights, etc.

    Returns: None (info is put into Tensorboard file).

    """
    if val_score is not None:
        writer.add_scalars('Loss', {'train': train_loss,
                                    'val': val_score},
                           epoch)
        logger.warning(f'\nValidation loss: {val_score}')

        # clearml_logger.report_scalar(title='rmsle', series='val', value=val_score, iteration=epoch + 1)
        # clearml_logger.report_scalar(title='rmsle', series='train', value=train_loss, iteration=epoch + 1)
    else:
        # clearml_logger.report_scalar('Loss', 'train', train_loss, epoch + 1)
        writer.add_scalar('Loss/train', train_loss, epoch + 1)

    logger.warning(f'\ntrain loss: {train_loss}')

    print_hist = cfg['validation']['hist']
    if print_hist:
        viz.vis_weight_dist(net, writer, epoch)
    logger.info('logging images...')
    fig = viz.show_batch(train_sample)
    fig.suptitle(f'train, epoch {epoch}', fontsize='xx-large')
    if cfg['misc']['plt_show']:
        plt.show()
    writer.add_figure(tag='viz/train', figure=fig, global_step=epoch)
    if val_sample is not None:
        # viz.get_sub_batch(val_sample, cfg['train']['batch_size'])
        fig = viz.show_batch(val_sample)
        fig.suptitle(f'val, epoch {epoch}', fontsize='xx-large')
        if cfg['misc']['plt_show']:
            plt.show()
        writer.add_figure(tag='viz/val', figure=fig, global_step=epoch)


def get_net():
    """
    get objects for training the network,
    as specified in configs.yml 'model'

    Returns: (criterion, net, optimizer) where:
        criterion: loss function for optimization.
        net: the network being used for training.
        optimizer:  optimization object (nn.optim)
    """
    cfg_model = cfg['model']
    cfg_checkpoint = cfg['checkpoint']
    cfg_optim = cfg['optim']
    model_name = cfg_model['name'].lower()
    if model_name == 'fcrn':
        net = FCRN()
    elif model_name == 'unet':
        # if cfg['dataset']['name'] == 'geopose':
        #     if cfg['dataset']['use_mask']:
        #         in_channel = 4
        #     else:
        #         in_channel = 3
        #     # TODO: fix if you don't do minmax scaling in the end.
        # else:
        #     in_channel = 3
        net = UNet(3)
    elif model_name.startswith('resnet'):
        net = model.ResnetUnet()
    elif model_name == 'toynet':
        net = model.toyNet()
    else:
        raise ValueError("model not supported.")
    if cfg_model['weight_init'] and not cfg_checkpoint['use_saved']:
        net.apply(weight_init)
        logger.info('init\'d weights with kaiming normal & zero bias')
    elif cfg_model['weight_file'] and model_name == 'fcrn':
        load_weights(net, cfg_model['weight_file'], torch.cuda.FloatTensor)
    net.to(device=get_dev())
    print('using ', get_dev())
    # TODO: use loss in configs for loss.
    criterion = get_criterion()
    optimizer = optim.Adam(net.parameters(), lr=cfg_optim['lr'])
    return criterion, net, optimizer


def get_criterion():
    cfg_optim = cfg['optim']
    loss_func_name = cfg_optim['loss'].lower()
    if loss_func_name.startswith('rmsle'):
        logger.info('using rmsle')
        criterion = model.RMSLELoss()
    elif loss_func_name.startswith('mse'):
        criterion = nn.MSELoss()
    elif loss_func_name == 'triple':
        criterion = model.triple_loss
    elif loss_func_name == 'grad':
        criterion = model.grad_loss
    elif loss_func_name == 'berhu':
        criterion = loss_berhu()
    else:
        raise ValueError("not good criterion.")
    return criterion


def get_geopose_split(subset_size, val_percent):
    if cfg['dataset']['use_mask']:
        tf = Compose([pad_and_center(),
                      prepare_data.ExtractSkyMask(),
                      prepare_data.ExtractSegmentationMaskSimple()])
    else:
        tf = pad_and_center()
    ds = GeoposeDataset(transform=tf)
    # ds = GeoposeDataset(transform=pad_and_center())
    # logger.warning('using hardcoded images, change when using non toy data!!')
    # imgs = ['eth_ch1_2011-04-30_18_37_52_01024',
    #         'eth_ch1_2011-04-30_18_40_20_01024']
    if subset_size is not None:
        # ds = Subset(ds, imgs)
        ds = Subset(ds, range(subset_size))
    n_val = int(len(ds) * val_percent)
    n_train = len(ds) - n_val
    train_split, val_split = random_split(ds,
                                          [n_train, n_val],
                                          generator=torch.Generator().manual_seed(42))
    return train_split, val_split


def get_loaders():
    """
    get data loaders for train set and val set

    Returns: train_loader and val_loader (pytorch dataloaders).

    """
    cfg_train = cfg['train']
    cfg_validation = cfg['validation']
    batch_size = cfg_train['batch_size']
    batch_size_val = cfg['validation']['batch_size']
    ds_name = cfg['dataset']['name']
    subset_size = cfg_train['subset_size']
    val_percent = cfg_validation['val_percent']
    if not val_percent:
        val_percent = 0
    if ds_name == 'farsight':
        train_split, val_split = get_farsight_split(cfg_train, subset_size, val_percent)
    elif ds_name == 'geopose':
        train_split, val_split = get_geopose_split(subset_size, val_percent)
    train_loader = DataLoader(train_split,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=0)
    val_loader = DataLoader(val_split,
                            shuffle=False,
                            batch_size=batch_size_val,
                            num_workers=0)
    if ds_name == 'farsight':
        return train_loader, val_loader
    elif ds_name == 'geopose':
        return train_loader, val_loader, reverseMinMaxScale


def get_farsight_split(cfg_train, subset_size, val_percent):
    if cfg_train['use_folds']:
        train_split, val_split = get_farsight_fold_dataset(1)
        if subset_size is not None:
            train_size = int(subset_size * (1 - val_percent))
            val_size = int(subset_size * val_percent)
            train_split = Subset(train_split, range(train_size))
            val_split = Subset(val_split, range(val_size))
    else:
        # TODO: generalize dataset to any dataset (hills for example).
        ds = FarsightDataset(transform=FarsightToTensor())
        if subset_size is not None:
            ds = Subset(ds, range(subset_size))
        n_val = int(len(ds) * val_percent)
        n_train = len(ds) - n_val
        train_split, val_split = random_split(ds,
                                              [n_train, n_val],
                                              generator=torch.Generator().manual_seed(42))
    # TODO: check rnd. gen is consistent.
    # TODO: make optional to use manual seed or random at some point. (same for DL?)
    return train_split, val_split


def save_checkpoint(epoch, net, optimizer, running_loss):
    """
    save a checkpoint of the network for future use.
    location is defined in configs.yml file.
    Args:
        epoch: int.
        net: network object (both weights and model object base is saved).
        optimizer
    # TODO: check rnd. gen is consistent.: optimizer object (only weights are saved).
        running_loss: float, current loss (for possibly future use).

    Returns: None (checkpoint saved).

    """
    logger.info(f'\nsaving checkpoint at epoch {epoch}...')
    folder = get_folder_name()
    folder = os.path.join('../models', folder)
    filename = 'epoch_' + str(epoch).zfill(4) + '.pt'
    if not os.path.exists(folder):
        os.mkdir(folder)
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        logger.warning(f'not saving {full_path} as it already exists.')
        return
    torch.save({
        'model': net,
        'optimizer': optimizer,
        'epoch': epoch,
        'loss': running_loss
    }, full_path)


def load_checkpoint(path=None):
    """
    load a saved checkpoint from the training process,
    be it for continued training or inference.
    location from which to load checkpoint is defined in configs.yml file.

    Args:
        path (string, optional): specify manual location of checkpoint.
                                 defaults to path in configs file.
    Returns: (net, optim, epoch, loss) where:
        net: weighted net
        optim: weighted optimizer
        epoch: updated epoch from checkpoint
        loss: current loss in checkpoint.

    """
    if path is None:
        path = os.path.join('..', 'models', cfg['checkpoint']['saved_path'])
    if not path.endswith('.pt'):
        # default to last trained model.
        path = os.path.join(path, max(os.listdir(path)))
    if not os.path.exists(path):
        raise FileNotFoundError
    logger.info(f'loading from {path}')
    checkpoint = torch.load(path)
    net = checkpoint['model']
    optim = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return net, optim, epoch, loss


def coarse_to_fine_train():
    lrs = [1e-4, 1e-2]
    ids = ['4f8b87a1e1684be9a8e34ede211d3233',
           '3e252824dbf0485ca4d16ce6e5daad88']
    for i, lr in enumerate(lrs):
        taskid = ids[i]
        cfg['optim']['lr'] = lr
        cfg['checkpoint']['run_name'] = f'optim_lr_{lr}'
        if cfg['misc']['use_trains']:
            clearml_logger, task = use_clearml(taskid)
        train()
        if cfg['misc']['use_trains']:
            task.close()


def use_clearml(taskid=None):
    if cfg['checkpoint']['use_saved']:
        cfg['checkpoint']['saved_path'] = cfg['checkpoint']['run_name']
        task = Task.init(continue_last_task=True,
                         reuse_last_task_id=taskid)
        task.set_initial_iteration(0)
        # task = Task.get_task(task_id='4f8b87a1e1684be9a8e34ede211d3233')
        # project_name='ariel-mde', task_name=get_folder_name())
    else:
        task = Task.init(project_name='ariel-mde', task_name=get_folder_name())
        config_file = task.connect_configuration(Path('configs.yml'), 'experiment_config')
        task_cfg = task.connect(cfg)  # enabling configuration override by clearml
        set_cfg(task_cfg)
    clearml_logger = task.get_logger()
    return clearml_logger, task


if __name__ == '__main__':
    """
    lr
    batch_size
    use_bn
    use_double_bn
    dropout
    random_seed
    flip_p
    """
    if cfg['misc']['use_trains']:
        _, task = use_clearml()
    train()
    if cfg['misc']['use_trains']:
        task.close()
    # UniformParameterRange('config/data_augmentation/horizontal_flip', min_value=0, max_value=1),
    # UniformParameterRange('config/data_augmentation/color_jitter', min_value=0, max_value=1),
    # UniformParameterRange('config/data_augmentation/gaussian_blur', min_value=0, max_value=1),
    # UniformParameterRange('config/data_augmentation/gaussian_noise', min_value=0, max_value=1)
    # train()
