import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


def toyNet():
    """just a simple net to get to learn pytorch."""
    return nn.Sequential(
        #         WeightValues('start'),
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        #         WeightValues('conv1'),
        nn.Conv2d(6, 4, 3),
        nn.ReLU(),
        #         WeightValues('conv2'),
        nn.ConvTranspose2d(4, 6, 3),
        nn.ReLU(),
        #         WeightValues('convT1'),
        nn.ConvTranspose2d(6, 1, 5),
        nn.ReLU(),
        #         WeightValues('convT2'),
        Squeeze()
    )


class ResnetUnet(nn.Module):
    def __init__(self, in_channels=3):
        """
        Unet with a pretrained Resnet34 backbone for encoding.
        outputs values in (0..1) range
        Args:
            in_channels:
        """
        super(ResnetUnet, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )

    def forward(self, image):
        x = self.model(image)
        return torch.sigmoid(x)


class WeightValues(nn.Module):
    """
    helper class to see mean and std at some layer.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print('weights at {}: mean: {}, std: {}'.format(self.name, x.mean(), x.std()))
        return x


if __name__ == '__main__':
    pass