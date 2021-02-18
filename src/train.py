import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
from clearml import Task
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

import losses_and_metrics
import model
import visualize as viz
from fcrn_pytorch.utils import loss_berhu
from other_arches.tiny_unet import UNet
from prepare_data import get_loaders
from utils import get_dev, cfg, get_folder_name, set_cfg, save_checkpoint, load_checkpoint
from fcrn_pytorch.fcrn import FCRN
from fcrn_pytorch.weights import load_weights

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
        None, saves checkpoints of net as it trains into file (+ is saved by clearml).
    """
    logger.info('getting params, dataloaders, etc...')
    cfg_train = cfg['train']
    cfg_checkpoint = cfg['checkpoint']
    cfg_validation = cfg['validation']
    epochs = cfg_train['epochs']
    print_every = cfg_train['print_every']
    save_every = cfg_checkpoint['save_every']
    folder_name = get_folder_name()
    writer = SummaryWriter(os.path.join('runs', folder_name))
    loaders = get_loaders()
    train_loader, val_loader = None, None
    len_loaders = len(loaders)
    if len_loaders == 4:
        train_loader, val_loader, test_loader, depth_postprocessing = loaders
    if len_loaders == 3:
        train_loader, val_loader, depth_postprocessing = loaders
    elif len_loaders == 2:
        train_loader, val_loader = loaders
        depth_postprocessing = None
    elif len_loaders == 1:
        train_loader = loaders[0]
        depth_postprocessing = None
    assert train_loader is not None and (
                val_loader is not None or not cfg_validation['val_round']), "problem with loader."
    n_batches = len(train_loader)
    cfg_model = cfg['model']
    cfg_checkpoint = cfg['checkpoint']
    cfg_optim = cfg['optim']
    if cfg_checkpoint['use_saved']:
        net, optimizer, epoch_start, running_loss = load_checkpoint()
        criterion = get_loss_function()
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
                if cfg['dataset']['use_mask'] and not cfg['dataset']['add_mask_to_image']:
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
                val_score, val_sample = eval_net(net, val_loader)
                scheduler.step(val_score)  # possibly plateau LR.
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(fr'old lr: {old_lr}, new lr: {new_lr}')
                old_lr = new_lr
            if epoch % print_every == print_every - 1:
                if not cfg_optim['use_lr_scheduler']:
                    if cfg_validation['val_round']:
                        assert cfg_validation[
                                   'val_percent'] is not None, 'required val_round but didn\'t give a split size'
                        val_score, val_sample = eval_net(net, val_loader)
                    else:
                        val_score = None
                        val_sample = None
                train_loss = running_loss / (print_every * n_batches)
                # TODO: see how to save og image for printing w.o doing it for every batch.
                train_sample = {**data, 'pred': pred_depth}
                if cfg['validation']['hist']:
                    viz_net = net
                else:
                    viz_net = None
                if depth_postprocessing:
                    logger.info('post-processing prediction and depth.')
                    train_sample = depth_postprocessing(train_sample)
                    if cfg_validation['val_round']:
                        val_sample = depth_postprocessing(val_sample)
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


def eval_net(net, loader):
    """
    Validation stage in the training loop.

    Args:
        net: network being trained
        loader: data loader of validation data
    Returns: score, sample_val: score of eval based on criterion + a sample from the eval_loader.

    """
    if cfg['validation']['metric'] is not None:
        metric = cfg['validation']['metric']
        assert metric in losses_and_metrics.__dict__, 'metric not implemented'
        metric = model.__dict__[metric]()
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
        net = UNet(3)
    elif model_name.startswith('resnet'):
        if cfg['dataset']['use_mask'] and cfg['dataset']['add_mask_to_image']:
            net = model.ResnetUnet(in_channels=4)
        else:
            net = model.ResnetUnet(in_channels=3)
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
    criterion = get_loss_function()
    optimizer = optim.Adam(net.parameters(), lr=cfg_optim['lr'])
    return criterion, net, optimizer


def get_loss_function():
    """
    Returns:

    """
    cfg_optim = cfg['optim']
    loss_func_name = cfg_optim['loss'].lower()
    if loss_func_name.startswith('rmsle'):
        logger.info('using rmsle')
        criterion = losses_and_metrics.RMSLELoss()
    elif loss_func_name.startswith('mse'):
        criterion = nn.MSELoss()
    elif loss_func_name == 'triple':
        criterion = losses_and_metrics.triple_loss
    elif loss_func_name == 'grad':
        criterion = losses_and_metrics.grad_loss
    elif loss_func_name == 'berhu':
        criterion = loss_berhu()
    else:
        raise ValueError("not good criterion.")
    return criterion


def coarse_to_fine_train():
    """
    super basic manual coarse-to-fine training.
    Returns:
        None
    """
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
    """
    does setup for clearml connection.
    Args:
        taskid: id of experiment to be reused. default is a new experiment.

    Returns: clearml_logger, task object.

    """
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
    if cfg['misc']['use_trains']:
        _, task = use_clearml()
    train()
    if cfg['misc']['use_trains']:
        task.completed()
