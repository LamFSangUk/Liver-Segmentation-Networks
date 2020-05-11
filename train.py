import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

import argparse
from argparse import RawTextHelpFormatter

from networks.Vnet import Vnet

from AbdomenDataset import AbdomenDataset

# define min loss for finding best model
min_loss = float("inf")


def train(model,
          epoch,
          data_loader,
          criterion,
          optimizer,
          scheduler,
          device,
          args):

    global min_loss
    total_loss = 0.0

    print('\n==> epoch %d' % epoch)

    model.train()

    for batch_idx, data in enumerate(data_loader):
        imgs = data['image'].to(device)
        imgs = imgs.unsqueeze(1)

        labels = data['label'].to(device)
        labels = torch.flatten(labels)
        # labels = labels.unsqueeze(0).view((2, -1))
        labels = labels.type(torch.int64)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Get output of model
        out = model(imgs)

        loss = criterion(out, labels)

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

    # Save checkpoint
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

    model = Vnet()
    model.to(device)

    model = nn.DataParallel(model).to(device)

    # criterion
    criterion = nn.NLLLoss().to(device)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, verbose=True, eps=1e-8)

    train_ds = AbdomenDataset("liver",
                              128, 128, 64,
                              path_image_dir="E:/Data/INFINITT/Integrated/train/img",
                              path_label_dir="E:/Data/INFINITT/Integrated/train/label")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)

    for epoch in range(500):
        train(model,
              epoch,
              train_loader,
              criterion,
              optimizer,
              scheduler,
              device,
              args)


if __name__ == "__main__":
    main()