import torch
import torch.nn as nn
import torch.nn.functional as F

from midl.layers.ActFunc import ActFunc
from midl.layers.losses import DiceLoss


class VoxResModule(nn.Module):
    def __init__(self, in_channels):
        super(VoxResModule, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.act1 = ActFunc('ReLU')
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm3d(in_channels)
        self.act2 = ActFunc('ReLU')
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.act1(self.bn1(x))
        out = self.conv1(out)

        out = self.act2(self.bn2(out))
        out = self.conv2(out)

        out += x

        return out


class VoxResNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(VoxResNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.act1 = ActFunc('ReLU')

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.act2 = ActFunc('ReLU')

        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.mod1 = VoxResModule(in_channels=64)
        self.mod2 = VoxResModule(in_channels=64)
        self.bn3 = nn.BatchNorm3d(64)
        self.act3 = ActFunc('ReLU')

        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.mod3 = VoxResModule(in_channels=64)
        self.mod4 = VoxResModule(in_channels=64)
        self.bn4 = nn.BatchNorm3d(64)
        self.act4 = ActFunc('ReLU')

        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.mod5 = VoxResModule(in_channels=64)
        self.mod6 = VoxResModule(in_channels=64)

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=n_classes, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=n_classes, kernel_size=4, stride=4)
        self.deconv4 = nn.ConvTranspose3d(in_channels=64, out_channels=n_classes, kernel_size=8, stride=8)

        self.softmax = F.softmax

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(self.bn1(out))

        out = self.conv2(out)
        c1 = out
        out = self.act2(self.bn2(out))

        out = self.conv3(out)
        out = self.mod1(out)
        out = self.mod2(out)
        c2 = out
        out = self.act3(self.bn3(out))

        out = self.conv4(out)
        out = self.mod3(out)
        out = self.mod4(out)
        c3 = out
        out = self.act4(self.bn4(out))

        out = self.conv5(out)
        out = self.mod5(out)
        out = self.mod6(out)
        c4 = out

        c1 = self.deconv1(c1)
        c2 = self.deconv2(c2)
        c3 = self.deconv3(c3)
        c4 = self.deconv4(c4)

        out = c1 + c2 + c3 + c4

        if self.training:
            self.out_for_loss = [out, c1, c2, c3, c4]

        return out

    def compute_loss(self, target, weight=(1.0, 1.0)):
        assert self.training is True

        c0, c1, c2, c3, c4 = self.out_for_loss

        c0 = self.softmax(c0, dim=1)
        c1 = self.softmax(c1, dim=1)
        c2 = self.softmax(c2, dim=1)
        c3 = self.softmax(c3, dim=1)
        c4 = self.softmax(c4, dim=1)

        metric = DiceLoss()

        loss = metric(c0, target, weight)
        loss += metric(c1, target, weight)
        loss += metric(c2, target, weight)
        loss += metric(c3, target, weight)
        loss += metric(c4, target, weight)

        return loss


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoxResNet(in_channels=1, n_classes=2)

    model = model.to(device)

    from torchsummary import summary
    summary(model, input_size=(1, 64, 128, 128))
    print(model)