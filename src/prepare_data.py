import numpy as np
import torch
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
import utils as defs
import visualize as viz
from data_loader import GeoposeDataset, GeoposeToTensor
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
# segmentation imports
import os, csv, scipy.io
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from torch.nn import functional as F


class ResizeToImgShape:
    def __call__(self, sample):
        """
        resize every meta-image (like depth, segmentation, etc.)
        to be same resolution as main image.
        Returns: sample

        Args:
            sample: dict(Tensors), data sample with an 'image', 'depth' and possibly other images
        """
        d1, d2, d3 = sample['image'].shape
        for k, img in sample.items():
            if k == 'image':
                continue
            if torch.is_tensor(img) or isinstance(img, Image.Image):
                # using Nearest, bc we want values to stay mostly the same,
                # and it's worse to have -.8 than some out of place -1's (for sky values).
                # segmap really can't be interpolated, bc values are supposed to be constant.
                img = TF.resize(img, (d2, d3), interpolation=Image.NEAREST)
                sample[k] = img
                # viz.tensor_imshow(img)
            elif type(img) == np.ndarray:
                img = cv2.resize(img, (d2, d1), interpolation=cv2.INTER_NEAREST)
                sample[k] = img
        # blended = blend_images(TF.to_pil_image(image), TF.to_pil_image(depth_resized))
        # plt.imshow(blended)
        # plt.show()
        return sample


class CropToAspectRatio:
    """
    crop sample images to required aspect ratio.
    the image then can just be rescaled to the required resolution.
    assumes all tensors are 3d, and grayscale images are of shape 1xHxW.
    """

    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def __call__(self, sample):
        for k, img in sample.items():
            if torch.is_tensor(img):
                # viz.tensor_imshow(img)
                h, w = img.shape[1], img.shape[2]
                if h * self.aspect_ratio > w:
                    disparity = int(h - w // self.aspect_ratio)
                    top = disparity // 2
                    height = h - disparity
                    img = TF.crop(img, top=top, left=0, height=height, width=w)
                    # viz.tensor_imshow(img)
                    # assert int(img.shape[1] * self.aspect_ratio) == w, 'aspect ratio still not checking out.'
                    sample[k] = img
                elif h * self.aspect_ratio < w:
                    disparity = int(w - h * self.aspect_ratio)
                    left = disparity // 2
                    width = w - disparity  # //2*2 to fix possible misalignment.
                    img = TF.crop(img, top=0, left=left, height=h, width=width)
                    # viz.tensor_imshow(img)
                    h, w = img.shape[1], img.shape[2]
                    assert int(h * self.aspect_ratio) == w, 'aspect ratio still not checking out.'
                    sample[k] = img
        return sample


def maybe_add_mask(sample):
    if defs.cfg['dataset']['use_mask']:
        assert 'mask' in sample, 'use_mask set to True but no mask provided in sample.'
        mask = sample['mask']
        sample['image'] = torch.cat((sample['image'], mask), dim=0)
    return sample


class ResizeToAlmostResolution:
    """
    resize image to closest resolution without stretching image.
    """

    def __init__(self, height, width, upper_bound=False):
        """
        settings for resizing
        Args:
            height:
            width:
            upper_bound: use upper bound of image (for cropping).
        """
        self.h = height
        self.w = width
        self.ar = self.w / self.h
        self.upper_bound = upper_bound

    def __call__(self, sample):
        for k, img in sample.items():
            if isinstance(img, np.ndarray):
                h, w = img.shape[0], img.shape[1]
                ar = w / h
                if (self.ar > ar and not self.upper_bound) or (self.ar < ar and self.upper_bound):
                    new_h, new_w = self.h, int(self.h * ar)
                elif (self.ar < ar and not self.upper_bound) or (self.ar > ar and self.upper_bound):
                    new_h, new_w = int(self.w / ar), self.w
                else:
                    new_h, new_w = self.h, self.w
                if k == 'image':
                    sample[k] = np.array(
                        TF.resize(Image.fromarray(img), (new_h, new_w), interpolation=Image.BILINEAR))
                else:
                    sample[k] = cv2.resize(img, (new_w, new_h),
                                           interpolation=cv2.INTER_NEAREST)  # don't want -1's and shit to get f'd up.
            elif torch.is_tensor(img):
                # viz.tensor_imshow(img)
                if k == 'image':
                    sample[k] = F.interpolate(img, (self.h, self.w),
                                              mode='bilinear')
                else:
                    sample[k] = TF.resize(img, (self.h, self.w),
                                          interpolation=Image.NEAREST)  # don't want -1's and shit to get f'd up.
                    # TODO: assert that values of images didn't change.
                # viz.tensor_imshow(sample[k])
        return sample


class ResizeToResolution:
    """
    resize all images in sample to be set resolution.
    Interpolation:
        - 'image' is resized using interpolation
        - all other images just fill with nearest value.
    to be used after cropping to aspect ratio with CropToAspectRatio.
    """

    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, sample):
        for k, img in sample.items():
            if isinstance(img, np.ndarray):
                if k == 'image':
                    sample[k] = np.array(
                        TF.resize(Image.fromarray(img), (self.h, self.w), interpolation=Image.BILINEAR))
                else:
                    sample[k] = cv2.resize(img, (self.w, self.h),
                                           interpolation=cv2.INTER_NEAREST)  # don't want -1's and shit to get f'd up.
            elif torch.is_tensor(img):
                # viz.tensor_imshow(img)
                if k == 'image':

                    sample[k] = F.interpolate(img, (self.h, self.w),
                                              mode='bilinear')
                else:
                    sample[k] = TF.resize(img, (self.h, self.w),
                                          interpolation=Image.NEAREST)  # don't want -1's and shit to get f'd up.
                    # TODO: assert that values of images didn't change.
                # viz.tensor_imshow(sample[k])
        return sample


class PadToAspectRatio:
    def __init__(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def __call__(self, sample):
        for k, img in sample.items():
            if torch.is_tensor(img) or isinstance(img, np.ndarray):
                # TODO: testing to see all images were originally in same resolution.
                if isinstance(img, np.ndarray):
                    sample_h, sample_w = img.shape[0], img.shape[1]
                else:
                    sample_h, sample_w = img.shape[1], img.shape[2]
                sample_aspect_ratio = sample_w / sample_h
                pad_h, pad_w = 0, 0
                if sample_aspect_ratio < self.aspect_ratio:
                    ar_disparity = self.aspect_ratio - sample_aspect_ratio
                    pad_w = ar_disparity * sample_h
                elif sample_aspect_ratio > self.aspect_ratio:
                    ar_disparity = sample_aspect_ratio - self.aspect_ratio
                    pad_h = ar_disparity * sample_h
                # add 1 extra pad for right and bottom if resolution diff is odd.
                border = (int(pad_w / 2), int(pad_h / 2), int((pad_w + 1) / 2), int((pad_h + 1) / 2))
                if isinstance(img, np.ndarray):
                    #  top, bottom, left, right
                    sample[k] = np.array(TF.pad(img, border))
                else:
                    sample[k] = TF.pad(img, border)
                    # sample[k] = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
                # else:
                # left, top, right and bottom

        return sample


class PadToResolution:
    """
    Pad all images in sample to meet required resolution. Pretty simple.
    """

    def __init__(self, height, width):
        # self.h = height
        # self.w = width
        self.pad_to_aspect_ratio = PadToAspectRatio(width / height)

    def __call__(self, sample):
        self.pad_to_aspect_ratio = PadToAspectRatio(sample)
        # for k, img in sample.items():
        #     if torch.is_tensor(img) or isinstance(img, np.ndarray):
        #         # viz.tensor_imshow(img)
        #         if isinstance(img, np.ndarray):
        #             sample_h, sample_w = img.shape[0], img.shape[1]
        #         else:
        #             sample_h, sample_w = img.shape[1], img.shape[2]
        #         pad_h, pad_w = 0, 0
        #         if sample_h < self.h:
        #             pad_h = self.h - sample_h
        #         if sample_w < self.w:
        #             pad_w = self.w - sample_w
        #         # pad to get to required resolution,
        #         # add 1 extra pad for right and bottom if resolution diff is odd.
        #         border = (pad_w // 2, pad_h // 2, (pad_w + 1) // 2, (pad_h + 1) // 2)
        #         if isinstance(img, np.ndarray):
        #             sample[k] = cv2.copyMakeBorder(img, border)
        #         else:
        #             sample[k] = TF.pad(img, border)
        #         # viz.tensor_imshow(sample[k])
        # return sample


class CenterAndCrop:
    def __init__(self, height, width):
        """
        crop out center of image to be in resolution hxw.

        Args:
            height: height (x)
            width: width (y)
        """
        self.h = height
        self.w = width

    def __call__(self, sample):
        for k, v in sample.items():
            if torch.is_tensor(v):
                # viz.tensor_imshow(v)
                sample[k] = TF.center_crop(v, (self.h, self.w))
                # viz.tensor_imshow(sample[k])
                assert sample[k].shape[1:3] == (self.h, self.w), f"problem with centering for {sample['name']}"
        return sample


class ExtractSkyMask:
    def __call__(self, sample):
        """
        IMPORTANT: mask extraction must be done BEFORE normalization!

        Objective: mask out the sky (as it just confuses the model).
        In any case, we don't care about it.

        Args:
            sample: Geopose dataset sample, possibly already containing a 'mask'.

        Returns: sample with masked out sky added in 'mask' key (i.e.
                (added to existing mask, if exists).
        """
        depth = sample['depth']
        # (-1 - cfg['normalization']['depth_mean'] / cfg['normalization']['depth_std'])
        if depth.min() == -1:
            sky_mask = depth != -1
        elif depth.min() == 0:
            sky_mask = depth != 0
        else:
            sky_mask = torch.full_like(depth, True, dtype=torch.bool)

        if 'mask' in sample:
            current_mask = sample['mask']
            sample['mask'] = current_mask & sky_mask
        else:
            sample['mask'] = sky_mask
        assert sample['mask'].dtype == torch.bool, 'mask contains non-binary values'
        return sample


class ExtractSegmentationMask:
    def __init__(self):
        """
        extract semantic map from pretrained model, for knowing where to ignore in the image,
        since we only have depth info on mountains..
        """
        self.names = {}
        with open('semseg/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

        # Network Builders
        self.net_encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='semseg/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
        self.net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=2048,
            num_class=150,
            weights='semseg/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
            use_softmax=True)

        self.crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(self.net_encoder, self.net_decoder, self.crit)
        self.segmentation_module.eval()
        self.segmentation_module.to(device=defs.get_dev())

    def visualize_result(self, img, pred, index=None):
        # filter prediction class if requested
        colors = scipy.io.loadmat('semseg/color150.mat')['colors']
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
            print(f'{self.names[index + 1]}:')

        # colorize prediction
        pred_color = colorEncode(pred, colors).astype(np.float32) / 255

        # aggregate images and save
        im_vis = np.concatenate((img, pred_color), axis=1)
        plt.imshow(im_vis)
        plt.show()

    def __call__(self, sample):
        img_data = sample['image']
        img_original = sample['og_image'].cpu().numpy().transpose(1, 2, 0)
        print(img_data.shape)
        singleton_batch = {'img_data': img_data[None]}
        output_size = img_data.shape[1:]
        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch, segSize=output_size)

        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()
        self.visualize_result(img_original, pred)
        # Top classes in answer
        # predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
        # for c in predicted_classes[:15]:
        #     self.visualize_result(img_original, pred, c)
        plt.show()
        return {**sample, 'segmask': TF.to_tensor(pred)}


class ExtractSegmentationMaskSimple:
    """same idea as non-simple counterpart, only uses a simpler model.
    Categories are as follows (we only care about background):
    {0: '__background__',
     1: 'aeroplane',
     2: 'bicycle',
     3: 'bird',
     4: 'boat',
     5: 'bottle',
     6: 'bus',
     7: 'car',
     8: 'cat',
     9: 'chair',
     10: 'cow',
     11: 'diningtable',
     12: 'dog',
     13: 'horse',
     14: 'motorbike',
     15: 'person',
     16: 'pottedplant',
     17: 'sheep',
     18: 'sofa',
     19: 'train'}
"""

    def __init__(self):
        self.segmodel = deeplabv3_resnet50(pretrained=True).to(device=defs.get_dev())
        self.segmodel.eval()

    def __call__(self, sample):
        segmap = self.segmodel(sample['image'].unsqueeze(0))['out'][0]
        segmap_pred = segmap.argmax(0)
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        # plot the semantic segmentation predictions of 21 classes in each color
        h, w = list(sample['image'].shape[1:])
        # r = Image.fromarray(segmap_pred.byte().cpu().numpy()).resize((w, h))
        # r.putpalette(colors)
        # plt.subplot(121)
        # plt.imshow(r)
        # plt.subplot(122)
        # viz.tensor_imshow(sample['og_image'])
        # plt.show()
        segmentation_mask = segmap_pred == 0
        if 'mask' in sample:
            sample['mask'] = sample['mask'] & segmentation_mask
        else:
            sample['mask'] = segmentation_mask
        return sample
        # viz.show_sample({**sample, 'segmap_pred': segmap_pred})


class NormalizeDepth:
    def __call__(self, sample):
        depth_mean = torch.Tensor([cfg['normalization']['depth_mean']])
        depth_std = torch.Tensor([cfg['normalization']['std_mean']])
        sample['depth'] = TF.normalize(sample['depth'], depth_mean, depth_std).to(device=defs.get_dev())
        # adding min to norm so all values will be above 0 (since we're using rmsle).
        if cfg['optim']['loss'].lower() == 'rmsle':
            sample['depth'] += torch.Tensor(depth_mean - 1 / depth_std)
        return sample


def norm_img_imagenet(sample):
    sample['og_image'] = sample['image'].clone()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # x['image'] = (x['image'] / 255 - mean) / std
    sample['image'] = TF.normalize(sample['image'], mean=mean, std=std)
    return sample


class NormalizeImg:
    """normalize images and depths based on our crop results."""

    # {'image': tensor([0.4168, 0.4701, 0.5008]), 'depth': array([3808.661], dtype=float32)}
    # {'image': tensor([0.1869, 0.1803, 0.1935]), 'depth': array([3033.6562], dtype=float32)}
    def __call__(self, sample):
        image_mean = torch.Tensor(defs.cfg['normalization']['image_mean']).to(defs.get_dev())
        image_std = torch.Tensor(defs.cfg['normalization']['image_std']).to(defs.get_dev())
        # sample['original_image'] = sample['image']
        sample['image'] = TF.normalize(sample['image'], image_mean, image_std)
        return sample


class NormMinMaxDepth:
    """
    minmax scaling, see
    https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    """

    def __call__(self, sample):
        # epsilon = 1e-10
        min_val = defs.cfg['normalization']['depth_min']
        max_val = defs.cfg['normalization']['depth_max']  # found from EDA,
        depth = sample['depth']
        depth -= min_val
        depth /= max_val - min_val
        # depth += epsilon  # for ability to plot.
        assert depth.min() >= 0 and depth.max() <= 1, "depth is out of [0-1] range"
        sample['depth'] = depth
        return sample


def reverseMinMaxScale(img):
    """
    reverses the minmax scaling done to the depth image.
    Args:
        img: depth image to be rescaled to original depths

    Returns: rescaled image.

    """
    min_val = defs.cfg['normalization']['depth_min']
    max_val = defs.cfg['normalization']['depth_max']  # found from EDA,
    img *= max_val - min_val
    img += min_val
    return img


def pad_to_aspect_ratio_and_resize():
    cfg_ds = defs.cfg['dataset']
    aspect_ratio, h, w = cfg_ds['aspect_ratio'], cfg_ds['h'], cfg_ds['w']
    return transforms.Compose([FillNaNsFFS(),
                               GeoposeToTensor(),
                               PadToResolution])


def crop_to_aspect_ratio_and_resize():
    cfg_ds = defs.cfg['dataset']
    aspect_ratio, h, w = cfg_ds['aspect_ratio'], cfg_ds['h'], cfg_ds['w']
    return transforms.Compose([FillNaNsFFS(),
                               GeoposeToTensor(),
                               CropToAspectRatio(aspect_ratio=aspect_ratio),
                               ResizeToResolution(h, w),
                               norm_img_imagenet,
                               NormMinMaxDepth()])
    # ExtractSkyMask()])


def pad_and_center():
    cfg_ds = defs.cfg['dataset']
    h, w = cfg_ds['h'], cfg_ds['w']
    return transforms.Compose([FillNaNsFFS(),
                               ResizeToImgShape(),
                               ResizeToAlmostResolution(h, w, upper_bound=True),
                               GeoposeToTensor(),
                               CenterAndCrop(h, w),
                               norm_img_imagenet,
                               NormMinMaxDepth()])


class ToPIL:
    def __call__(self, sample):
        for k, img in sample.items():
            if isinstance(img, np.ndarray):
                sample[k] = Image.fromarray(img)
        return sample


class FillNaNsFFS:
    def __call__(self, sample):
        depth = sample['depth']
        sample['depth'] = np.nan_to_num(depth)
        return sample


if __name__ == '__main__':
    # geoset = GeoposeDataset(transform=transforms.Compose([GeoposeToTensor(),
    #                                                       ResizeDepth(),
    #                                                       CenterAndCrop(680, 1024)
    #                                                       ]))
    # geoset_no_crop = GeoposeDataset(transform=transforms.Compose([GeoposeToTensor(),
    #                                                               ResizeDepth()]))
    # dl = DataLoader(geoset, batch_size=4)
    # batch = next(iter(dl))
    # s2 = next(iter(geoset_no_crop))
    # print single sample,
    # since if we don't crop we can't have them all in one batch.
    cfg_ds = defs.cfg['dataset']
    aspect_ratio, h, w = cfg_ds['aspect_ratio'], cfg_ds['h'], cfg_ds['w']

    shitty_transform = transforms.Compose([FillNaNsFFS(),
                                           GeoposeToTensor(),
                                           ResizeToImgShape(),
                                           ResizeToResolution(h, w)])
    geoset = GeoposeDataset(transform=transforms.Compose([
        FillNaNsFFS(),
        ResizeToImgShape(),
        ResizeToAlmostResolution(180, 240),
        GeoposeToTensor(),
        ExtractSkyMask(),
        norm_img_imagenet,
        ExtractSegmentationMaskSimple(),
        # TODO: set_seg_to_null or minus one, whatever.
        PadToAspectRatio(4 / 3),

    ]))
    # flickr_sge_13078985295_c4204c63a7_o
    # 28488116812_f5a57ca0f6_k
    # flickr_sge_10118947555_a8a13b6338_o_grid_1_0
    # .008_0
    # .008.xml_0_1_0
    # .682226
    # flickr_sge_12052835295_6bc07b9b18_o
    # eth_ch1_IMG_5238_01024
    # flickr_sge_12052835295_6bc07b9b18_o
    sample = geoset['flickr_sge_3476145360_a989526084_o']
    # viz.show_batch(batch)
    viz.show_sample(sample)
    plt.show()
