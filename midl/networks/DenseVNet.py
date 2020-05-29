import torch
import torch.nn as nn
import torch.nn.functional as F

from midl.layers.ActFunc import ActFunc
from midl.layers.Densenet.DenseBlockCompressed import DenseBlockCompressed as DenseBlock
from midl.layers.losses import DiceLoss


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = ActFunc('ReLU')

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_channels, drop_rate):
        super(DenseLayer, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=growth_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm3d(growth_channels)
        self.act = ActFunc('ReLU')

        self.dropout = F.dropout3d
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        out = self.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], 1)


# class OutputTransition(nn.Module):
#     def __init__(self, in_channels):
#         super(OutputTransition, self).__init__()
#
#         self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=2, kernel_size=5, padding=2, bias=False)
#         self.bn1 = nn.BatchNorm3d(2)
#         self.conv2 = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=1, bias=False)
#         self.act1 = ActFunc('PReLU', num_parameters=2)
#         self.softmax = F.softmax
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act1(out)
#
#         out = self.conv2(out)
#
#         # make channels the last axis
#         out = out.permute(0, 2, 3, 4, 1).contiguous()
#         out = out.view(out.numel() // 2, 2)
#         out = self.softmax(out, dim=1)
#
#         # treat channel 0 as the predicted output
#         return out


class DownTransition(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownTransition, self).__init__()

        self.down = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = ActFunc('ReLU')
        self.dropout = F.dropout3d

    def forward(self, x):
        out = self.down(x)
        out = self.dropout(out, p=0, training=self.training)
        return out


class DenseVNet(nn.Module):
    def __init__(self, in_channels, shape, n_classes):
        super(DenseVNet, self).__init__()

        self.n_classes = n_classes

        self.feature = DownTransition(in_channels=in_channels,
                                      out_channels=24,
                                      kernel_size=5,
                                      stride=2,
                                      padding=2)

        self.dfs1 = DenseBlock(nb_layers=5,
                               in_channels=24,
                               growth_rate=4,
                               block=DenseLayer,
                               drop_rate=0)
        self.down12 = DownTransition(in_channels=20,
                                     out_channels=24,
                                     kernel_size=3,
                                     stride=1)

        self.dfs2 = DenseBlock(nb_layers=10,
                               in_channels=24,
                               growth_rate=8,
                               block=DenseLayer,
                               drop_rate=0)
        self.down23 = DownTransition(in_channels=80,
                                     out_channels=24,
                                     kernel_size=3,
                                     stride=1)

        self.dfs3 = DenseBlock(nb_layers=10,
                               in_channels=24,
                               growth_rate=16,
                               block=DenseLayer,
                               drop_rate=0)

        self.skip1 = ConvUnit(in_channels=20, out_channels=12)
        self.skip2 = ConvUnit(in_channels=80, out_channels=24)
        self.skip3 = ConvUnit(in_channels=160, out_channels=24)
        self.up1 = nn.Upsample(size=shape, mode="trilinear")
        self.up2 = nn.Upsample(size=shape, mode="trilinear")
        self.up3 = nn.Upsample(size=shape, mode="trilinear")

        self.conv = ConvUnit(in_channels=12+24+24,
                             out_channels=n_classes)

        self.prior = nn.Parameter(torch.randn(1, 1, shape[0]//12, shape[1]//12, shape[2]//12))
        self.up_prior = nn.Upsample(size=shape, mode="trilinear")

        self.softmax = F.softmax

    def forward(self, x):
        out = self.feature(x)

        out = self.dfs1(out)
        skip1 = self.skip1(out)

        out = self.down12(out)
        out = self.dfs2(out)
        skip2 = self.skip2(out)

        out = self.down23(out)
        out = self.dfs3(out)
        skip3 = self.skip3(out)

        skip1 = self.up1(skip1)
        skip2 = self.up2(skip2)
        skip3 = self.up3(skip3)
        skip = torch.cat([skip1, skip2, skip3], 1)
        print(skip.shape)
        out = self.conv(skip)

        prior = self.up_prior(self.prior)
        print(out.shape)
        print(prior.shape)
        out = out + prior

        if self.training:
            self.out_for_loss = out

        return out

    def compute_loss(self, target, weight=(1.0, 1.0)):
        assert self.training is True

        out = self.out_for_loss
        out = F.softmax(out, dim=1)

        metric = DiceLoss()

        loss = metric(out, target, weight)

        return loss

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseVNet(in_channels=1, shape=(64, 128, 128), n_classes=2)

    model = model.to(device)

    from torchsummary import summary
    summary(model, input_size=(1, 64, 128, 128))
    print(model)