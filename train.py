import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import os

import argparse
from argparse import RawTextHelpFormatter

import pickle

import midl

# define min loss for finding best model
min_loss = float("inf")


def train(model,
          epoch,
          data_loader,
          criterion,
          optimizer,
          scheduler,
          writer,
          device,
          args):

    global min_loss
    total_loss = 0.0

    print('\n==> epoch %d' % epoch)

    model.train()

    for batch_idx, data in enumerate(data_loader):
        imgs = data['image'].to(device)
        imgs = imgs.float()
        imgs = imgs.unsqueeze(1)

        labels = data['label'].to(device)
        labels = torch.flatten(labels)
        labels = labels.type(torch.int64)

        # Calculate weights
        target_mean = data_loader.dataset.label_mean
        bg_weights = target_mean / (1. + target_mean)
        fg_weight = 1. - bg_weights
        class_weights = torch.from_numpy(np.array([bg_weights, fg_weight])).float()
        class_weights = class_weights.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Get output of model
        out = model(imgs)

        print(out)

        # Loss calculation
        # NLL loss
        # loss = F.nll_loss(out, labels, weight=class_weights)

        # Dice loss
        labels = F.one_hot(labels, num_classes=2)
        loss = midl.layers.losses.DiceLoss(weight=class_weights)(out, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Progress bar
        data_cnt = len(data_loader.dataset)
        done_cnt = min((batch_idx + 1) * data_loader.batch_size, data_cnt)
        rate = done_cnt / data_cnt
        bar = ('=' * int(rate * 32) + '>').ljust(32, '.')
        idx = str(done_cnt).rjust(len(str(data_cnt)), ' ')
        print('\rTrain\t : {}/{}: [{}]'.format(
            idx,
            data_cnt,
            bar
        ), end='')

    print()
    print("Total Loss: %f"  % total_loss)
    print()

    # Logging with tensorboard
    writer.add_scalars("Loss/train", {
        "loss": total_loss,
    }, epoch+1)

    # Save checkpoint
    if epoch % 10 == 0:
        state = {
            'epoch': epoch,
            'net': model.state_dict()
        }
        torch.save(state, os.path.join(args.model_save_path, "model-%d.pth"%epoch))

    if total_loss < min_loss:
        state = {
            'epoch': epoch,
            'net': model.state_dict()
        }
        torch.save(state, os.path.join(args.model_save_path, "best.pth"))

        min_loss = total_loss

    scheduler.step(total_loss)


def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model_save_path',
                        default='E:/Data/INFINITT/Models',
                        help='Directory path to save model checkpoints')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # criterion
    criterion = nn.NLLLoss().to(device)
    # criterion = DiceLoss.apply
    # criterion = DiceLoss()

    # Vnet
    model = midl.networks.Vnet()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, verbose=True, eps=1e-10)

    # VoxResNet
    # model = midl.networks.VoxResNet(in_channels=1, n_classes=2)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, verbose=True, eps=1e-10)

    model.to(device)

    model = nn.DataParallel(model).to(device)

    # train_ds = midl.ds.AbdomenDataset("liver",
    #                           128, 128, 64,
    #                           path_image_dir="E:/Data/INFINITT/Integrated/train/img",
    #                           path_label_dir="E:/Data/INFINITT/Integrated/train/label")
    with open("./utils/train_ds", "rb") as f:
        train_ds = pickle.load(f)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True,
                                               num_workers=6,
                                               pin_memory=True)

    # Tensorboard Logging
    writer = SummaryWriter('E:/Data/INFINITT/logs')

    for epoch in range(300):
        train(model,
              epoch,
              train_loader,
              criterion,
              optimizer,
              scheduler,
              writer,
              device,
              args)

    writer.close()


if __name__ == "__main__":
    main()