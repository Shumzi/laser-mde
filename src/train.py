import torch.nn as nn
import torch.nn.functional as F
import torch
from data_loader import FarsightDataset, ToTensor
from utils.definitions import DATA_DIR, get_depth_dir, get_img_dir
import visualize as viz
import model
import torch.optim as optim
import matplotlib.pyplot as plt
def train(epochs = 2):
    """
    main training loop.
    """
    print('started')
    ds = FarsightDataset(img_dir=get_img_dir(),
                         depth_dir=get_depth_dir(),
                         transform=ToTensor())
    indices = torch.randperm(len(ds))[:30]
    ds_sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(ds,
                                             sampler=ds_sampler,
                                             batch_size=10,
                                             num_workers=2)
    print(len(dataloader))
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['depth'].size())

        # observe 4th batch and stop.
        if i_batch == 0:
            plt.figure()
            viz.show_depths_batch(sample_batched)
            # plt.axis('off')
            plt.show()
            break

    # for data in enumerate():
    # net = model.ToyNet()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(net.parameters())
    for epoch in range(2):  # loop over the dataset multiple times
        break
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [input images, depth maps]
            inputs, gt_depths = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, gt_depths)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished Training')

train()
