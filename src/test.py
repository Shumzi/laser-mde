import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import visualize as viz
from depthestim import pred as ken_net
from losses_and_metrics import RMSLELoss, Accuracy, RMSLEScaleInvariantError, GradLoss
from prepare_data import get_loaders
from utils import load_checkpoint


def aggregate_batches(list_of_batches):
    """
    aggregates batches into one big batch
    Args:
        list_of_batches: list of batches

    Returns: aggregated batch.

    """
    agg_batch = {}
    for batch in list_of_batches:
        for k, v in batch.items():
            if type(v) is list:
                cur = agg_batch.get(k, [])
                agg_batch[k] = cur + v
            elif torch.is_tensor(v):
                if k not in agg_batch:
                    agg_batch[k] = v
                else:
                    agg_batch[k] = torch.cat((agg_batch[k], v), dim=0)
    return agg_batch


def get_only_relevant_images_in_batch(batch, scores):
    """
    throw away shitty images (normalized ones for example)
     and only keep the interesting ones in every batch.
    Args:
        batch: batch to take subset of images from
        scores:

    Returns: non-shitty batch.

    """
    batch = batch.copy()
    batch['image'] = batch.pop('og_image')
    batch['pred_my_net'] = batch.pop('pred')
    batch['ground_truth'] = batch.pop('depth')
    for i, name in enumerate(batch['name']):
        batch['name'][i] = f'{scores[i]:.4f}'

    return batch


def show_best(batches, scores, score_name, reverse=False):
    """
    show best batches according to score
    Args:
        batches: list of batches
        scores: list of scores for each batch
        score_name: name of score being shown
        reverse: makes best be highest value instead of lowest.

    Returns: None, fig of best batches in mpl.

    """
    if reverse:
        idxs = np.array(np.argsort(scores)[-5:][::-1])
    else:
        idxs = np.argsort(scores)[:5]
    viz.show_batch(get_only_relevant_images_in_batch(aggregate_batches(batches[idxs]), scores[idxs]))
    plt.suptitle(f'Best in {score_name}', fontsize='xx-large')
    plt.tight_layout()


def show_worst(batches, scores, score_name, reverse=False):
    """
    show worst batches according to score
    Args:
        batches: list of batches
        scores: list of scores for each batch
        score_name: name of score being shown
        reverse: makes worst be lowest value instead of highest.

    Returns: None, fig of worst batches in mpl.

    """
    if reverse:
        idxs = np.array(np.argsort(scores)[:5][::-1])
    else:
        idxs = np.argsort(scores)[-5:]
    viz.show_batch(get_only_relevant_images_in_batch(aggregate_batches(batches[idxs]), scores[idxs]))
    plt.suptitle(f'Worst in {score_name}', fontsize='xx-large')
    plt.tight_layout()


def test(net, score_only_masked_area=True, ken_net=False):
    """
    
    Args:
        net: forward function of net. 
        score_only_masked_area: only gives a score for areas not in the mask.
        ken_net: if using ken_net, set to true (doesn't add mask to image in this case, for example).

    Returns: scores, batches of test set + predictions..

    """
    rmsle = RMSLELoss()
    sie = RMSLEScaleInvariantError()
    grad_loss_metric = GradLoss()
    acc = Accuracy()
    loaders = get_loaders()
    test_loader = loaders[2]
    rescaler = loaders[3]()
    rmsle_loss_sum = 0
    d1_sum = 0
    d2_sum = 0
    d3_sum = 0
    si_sum = 0
    grad_loss_sum = 0
    rmsles = np.array([])
    d1s = np.array([])
    grads = np.array([])
    batches = np.array([])
    for batch in tqdm(test_loader):
        if ken_net:
            img = batch['og_image']
            img = img[:, :3]
            torch.cuda.empty_cache()
            batch = rescaler({**batch, 'pred': torch.zeros(1)})
            batch.pop('pred')
        else:
            img = batch['image']
        pred = net(img)  # [:, :3]  # since ken doesn't use the mask for prediction
        batch = {**batch, 'pred': pred}
        if not ken_net:
            batch = rescaler(batch)
        if score_only_masked_area:
            gt = batch['depth']
            mask = torch.logical_not(batch['mask'])
            masked_pred = pred[mask]
            masked_gt = gt[mask]
        else:
            gt = batch['depth']
            masked_pred = pred
            masked_gt = gt
        rmsle_loss = rmsle(masked_pred, masked_gt)
        d1 = acc(masked_pred, masked_gt)
        d2 = acc(masked_pred, masked_gt, num=2)
        d3 = acc(masked_pred, masked_gt, num=3)
        si = sie(masked_pred, masked_gt)
        grad_loss = grad_loss_metric(pred, gt)
        assert rmsle_loss >= 0 and not torch.isinf(rmsle_loss) and not torch.isnan(rmsle_loss), 'rmsle_loss'
        assert d1 >= 0 and d1 <= 1, 'd1'
        assert d2 >= 0 and d2 <= 1, 'd2'
        assert d3 >= 0 and d3 <= 1, 'd3'
        assert si >= 0 and not torch.isinf(si) and not torch.isnan(si), 'si'
        assert grad_loss >= 0 and not torch.isinf(grad_loss) and not torch.isnan(grad_loss), 'grad_loss'
        batches = np.append(batches, batch)
        rmsles = np.append(rmsles, rmsle_loss.item())
        d1s = np.append(d1s, d1.item())
        grads = np.append(grads, grad_loss.item())
        rmsle_loss_sum += rmsle_loss
        d1_sum += d1
        d2_sum += d2
        d3_sum += d3
        si_sum += si
        grad_loss_sum += grad_loss

    rmsle_loss_sum /= len(test_loader)
    d1_sum /= len(test_loader)
    d2_sum /= len(test_loader)
    d3_sum /= len(test_loader)
    si_sum /= len(test_loader)
    grad_loss_sum /= len(test_loader)
    print(f'rmsle_loss_sum: {rmsle_loss_sum:.4f}')
    print(f'd1_sum: {d1_sum:.4f}')
    print(f'd2_sum: {d2_sum:.4f}')
    print(f'd3_sum: {d3_sum:.4f}')
    print(f'si_sum: {si_sum:.4f}')
    print(f'grad_loss_sum: {grad_loss_sum:.4f}')
    return {'rmsles': rmsles, 'grads': grads, 'd1s': d1s}, batches


def test_my_net():
    """
    wrapper for test function to test my net (resnet-unet)
    Returns: None.

    """
    ckpt = load_checkpoint()
    net = ckpt[0]
    net.eval()
    history, batches = test(net=net)
    pickle.dump(history, open('my_net_hist.pkl', 'wb'))
    pickle.dump(batches, open('my_net_batches.pkl', 'wb'))


def test_ken_burns():
    """
    wrapper for test function to test ken net.
    Returns: None.

    """
    hist, batches = test(net=ken_net, ken_net=True)
    # pickle.dump(hist, open('ken_history_correct.pkl', 'wb'))
    # pickle.dump(batches, open('ken_batches_correct.pkl', 'wb'))


def merge_batches():
    """
    helper function to merge batch results of ken net and my net.
    Returns:

    """
    batches_my_net = pickle.load(open('my_net_batches.pkl', 'rb'))
    batches_ken = pickle.load(open('ken_batches_correct.pkl', 'rb'))
    for batch_my_net, batch_ken in tqdm(zip(batches_my_net, batches_ken), total=len(batches_my_net)):
        batch_my_net['ken_pred'] = batch_ken['pred']
    pickle.dump(batches_my_net, open('merged_batches_correct.pkl', 'wb'))


def save_best_and_worst_batches(batches, metric, title, reverse=False):
    """

    Args:
        batches:
        metric:
        title: title for figure, name of metric probably.
        reverse: reverses what's considered best and worst.

    Returns:

    """
    show_best(batches, metric, title, reverse)
    plt.savefig('../results/' + title + ' best.png')
    show_worst(batches, metric, title, reverse)
    plt.savefig('../results/' + title + ' worst.png')


def save_plot_results():
    """
    saves best and worst results from both nets.
    Returns: None

    """
    batches = pickle.load(open('merged_batches_correct.pkl', 'rb'))
    hist_my_net = pickle.load(open('my_net_hist.pkl', 'rb'))
    rmsle_my_net, d1s_my_net, grads_my_net = hist_my_net['rmsles'], hist_my_net['d1s'], hist_my_net['grads']
    hist_ken = pickle.load(open('ken_history.pkl', 'rb'))
    rmsle_ken, d1s_ken, grads_ken = hist_ken['rmsles'], hist_ken['d1s'], hist_ken['grads']
    save_best_and_worst_batches(batches, rmsle_my_net, 'RMSLE (my net)')
    save_best_and_worst_batches(batches, grads_my_net, 'Grads (my net)')
    save_best_and_worst_batches(batches, d1s_my_net, 'Delta1 (my net)', reverse=True)
    save_best_and_worst_batches(batches, rmsle_ken, 'RMSLE (ken net)')
    save_best_and_worst_batches(batches, grads_ken, 'Grads (ken net)')
    save_best_and_worst_batches(batches, d1s_ken, 'Delta1 (ken net)', reverse=True)


if __name__ == '__main__':
    # test_ken_burns()
    batches = pickle.load(open('merged_batches_correct.pkl', 'rb'))
    print('hi')
