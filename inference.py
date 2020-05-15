import torch
import torch.nn as nn

import argparse
from argparse import RawTextHelpFormatter

import nibabel as nib
import numpy as np

import os

import pickle

import midl


def inference(model,
              data_loader,
              device,
              args):

    model.eval()

    cnt = 0
    for data in data_loader:
        filename = data['name'][0]

        imgs = data['image'].to(device)
        imgs = imgs.float()
        imgs = imgs.unsqueeze(1)
        print(imgs.shape)

        output = model(imgs)
        print(output)
        _, output = output.max(1)
        print(output)
        print(np.unique(output.cpu().numpy()))

        output = output.view((64, 128, 128))
        output = output.cpu().numpy()
        output = np.transpose(output, axes=(1, 2, 0))
        output = output.astype(np.uint8)

        res = nib.Nifti1Image(output, np.eye(4))
        nib.save(res, os.path.join(args.path_res, '%s'%filename))

        cnt += 1


def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--path_res', default='E:/Data/INFINITT/Results/VoxResNet')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Vnet
    model = midl.networks.Vnet()

    # VoxResNet
    # model = VoxResNet(in_channels=1, n_classes=2)

    model.to(device)

    model = nn.DataParallel(model).to(device)

    checkpoint = torch.load('E:/Data/INFINITT/Models/best.pth')
    model.load_state_dict(checkpoint['net'], strict=False)

    # test_ds = AbdomenDataset("liver",
    #                          128, 128, 64,
    #                          path_image_dir="E:/Data/INFINITT/Integrated/test/img",
    #                          path_label_dir="E:/Data/INFINITT/Integrated/test/label")
    with open("./utils/test_ds", "rb") as f:
        test_ds = pickle.load(f)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False,
                                              num_workers=1)

    inference(model,
              test_loader,
              device,
              args)

if __name__ == "__main__":
    main()
