import torch.nn as nn
import torch.nn.functional as F
import torch
from data_loader import FarsightDataset, ToTensor
from utils.definitions import DATA_DIR, get_depth_dir, get_img_dir
import visualize as viz
import model
import torch.optim as optim
import matplotlib.pyplot as plt
from trains import Task

task = Task.init(project_name='mde', task_name='test loop')
logger = task.get_logger()


def train(epochs=2):
    """
    main training loop.
    """
    print('started')
    # create dataset
    ds = FarsightDataset(img_dir=get_img_dir(),
                         depth_dir=get_depth_dir(),
                         transform=ToTensor())
    indices = torch.randperm(len(ds))[:30]
    ds_sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(ds,
                                             sampler=ds_sampler,
                                             batch_size=4,
                                             num_workers=2)
    # TODO: fix weird float32 requirement in conv2d to work with uint8.
    net = nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.ConvTranspose2d(6, 1, 5),
        model.Squeeze()
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [input images, depth maps]
            inputs, gt_depths = data['image'], data['depth']

            optimizer.zero_grad()
            print('input shape {}, type: {}'.format(inputs.size(), inputs.dtype))
            outputs = net(inputs)
            print('out shape: {}, gt_shape: {}'.format(outputs.size(), gt_depths.size()))
            viz.show_batch({**data, 'pred': outputs.detach().numpy()})
            plt.show()
            loss = criterion(outputs, gt_depths)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
            break
    print('Finished Training')


train()
