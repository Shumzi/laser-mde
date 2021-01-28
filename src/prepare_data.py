import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

import utils as defs
import visualize as viz
from data_loader import GeoposeDataset, GeoposeToTensor


class ResizeToImgShape:
    def __call__(self, sample):
        """
        resize every meta-image (like depth, segmentation, etc.)
        to be same resolution as main image.
        Returns: sample

        Args:
            sample: dict(Tensors), data sample with an 'image', 'depth' and possibly other images
        """
        c, h, w = sample['image'].shape
        for k, img in sample.items():
            if k == 'image':
                continue
            if torch.is_tensor(img):
                # using Nearest, bc we want values to stay mostly the same,
                # and it's worse to have -.8 than some out of place -1's (for sky values).
                # segmap really can't be interpolated, bc values are supposed to be constant.
                img = TF.resize(img, (h, w), interpolation=Image.NEAREST)
                sample[k] = img
                # viz.tensor_imshow(img)
            elif type(img) == np.ndarray:
                img = resize(img, (c, h))
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
            if torch.is_tensor(img):
                # viz.tensor_imshow(img)
                if k == 'image':
                    sample[k] = TF.resize(img, (self.h, self.w))
                else:
                    sample[k] = TF.resize(img, (self.h, self.w),
                                          interpolation=Image.NEAREST)  # don't want -1's and shit to get f'd up.
                    # TODO: assert that values of images didn't change.
                # viz.tensor_imshow(sample[k])
        return sample


class PadToResolution:
    """
    Pad all images in sample to meet required resolution. Pretty simple.
    """

    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, sample):
        for k, img in sample.items():
            if torch.is_tensor(img):
                # viz.tensor_imshow(img)
                sample_h, sample_w = img.shape[1], img.shape[2]
                pad_h, pad_w = 0, 0
                if sample_h < self.h:
                    pad_h = self.h - sample_h
                if sample_w < self.w:
                    pad_w = self.w - sample_w
                # pad to get to required resolution,
                # add 1 extra pad for right and bottom if resolution diff is odd.
                sample[k] = TF.pad(img, (pad_w // 2, pad_h // 2, (pad_w + 1) // 2, (pad_h + 1) // 2))
                # viz.tensor_imshow(sample[k])
        return sample


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
        return sample


class ExtractSkyMask:
    def __call__(self, sample):
        """
        IMPORTANT: mask extraction must be done BEFORE normalization!

        Objective: mask out the sky (as it just confuses the model).
        In any case, we don't care about it.

        Args:
            sample: Geopose dataset sample, possibly already containing a 'mask'.

        Returns: sample with sky mask added in 'mask' key
                (added to existing mask, if exists).
        """
        depth = sample['depth']
        sky_mask = (depth != -1)
        if 'mask' in sample:
            current_mask = sample['mask']
            sample['mask'] = current_mask | sky_mask
        else:
            sample['mask'] = sky_mask
        return sample


class ExtractSegmentationMask:
    pass
    # TODO: next week, the whole segmentation git.


class NormalizeDepth:
    def __call__(self, sample):
        depth_mean = torch.Tensor([3808.661])
        depth_std = torch.Tensor([3033.6562])
        # adding min to norm so all values will be above 0 (since we're using rmsle).
        sample['depth'] = TF.normalize(sample['depth'], depth_mean, depth_std) + torch.Tensor(
            depth_mean - 1 / depth_std).to(device=defs.get_dev())
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


def crop_to_aspect_ratio_and_resize():
    cfg_ds = defs.cfg['dataset']
    aspect_ratio, h, w = cfg_ds['aspect_ratio'], cfg_ds['h'], cfg_ds['w']
    return transforms.Compose([FillNaNsFFS(),
                               GeoposeToTensor(),
                               CropToAspectRatio(aspect_ratio=aspect_ratio),
                               ResizeToResolution(h, w),
                               NormalizeImg(),
                               NormMinMaxDepth()])
    # ExtractSkyMask()])


def pad_and_center():
    cfg_ds = defs.cfg['dataset']
    h, w = cfg_ds['h'], cfg_ds['w']
    return transforms.Compose([FillNaNsFFS(),
                               GeoposeToTensor(),
                               ResizeToImgShape(),
                               PadToResolution(h, w),
                               CenterAndCrop(h, w),
                               NormalizeImg(),
                               NormMinMaxDepth()])


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
    geoset = GeoposeDataset(transform=transforms.Compose([ExtractSkyMask(),
                                                          pad_and_center()]))
    dl = DataLoader(geoset, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    # for k, v in s2.items():
    #     if torch.is_tensor(v):
    #         s2[k] = v.unsqueeze(0)
    #     elif type(v).__name__ == 'str_':
    #         s2[k] = [v]
    viz.show_batch(batch)
    plt.show()
    # viz.show_batch(s2)
    # plt.show()
