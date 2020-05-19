import torch
import torch.nn as nn
import torch.nn.functional as F

from midl.layers.ActFunc import ActFunc
from midl.layers.AttentionGate import AttentionGate

from midl.layers.Densenet.DenseBlockCompressed import DenseBlockCompressed as DenseBlock

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


def _make_conv_layer(in_channels, n_convs):
    layers = []
    for i in range(n_convs):

        # Adding non-linearity
        if i != 0:
            layers.append(ActFunc('PReLU', num_parameters=in_channels))

        layers.append(ConvBlock(in_channels=in_channels, out_channels=in_channels))

    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        return out


class InputTransition(nn.Module):
    def __init__(self):
        super(InputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.act1 = ActFunc('PReLU', num_parameters=16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = out + x
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=2, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=1, bias=False)
        self.act1 = ActFunc('PReLU', num_parameters=2)
        self.softmax = F.softmax

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out, dim=1)

        # treat channel 0 as the predicted output
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channels, n_convs):
        super(DownTransition, self).__init__()

        out_channels = 2 * in_channels
        self.down_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.act1 = ActFunc('PReLU', num_parameters=in_channels)

        #self.conv_block = _make_conv_layer(out_channels, n_convs)
        self.conv_block = DenseBlock(nb_layers=4,
                                     in_channels=in_channels,
                                     growth_rate=in_channels//4,
                                     block=DenseLayer,
                                     drop_rate=0)
        # self.act2 = ActFunc('PReLU', num_parameters=out_channels)


    def forward(self, x):
        x = self.down_conv(x)
        x = self.bn1(x)
        x = self.act1(x)

        out = self.conv_block(x)
        out = torch.cat([x, out], dim=1)
        # out = self.act2(out)

        return out


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs):
        super(UpTransition, self).__init__()

        self.de_conv = nn.ConvTranspose3d(in_channels=in_channels,
                                          out_channels=out_channels // 2,
                                          kernel_size=2,
                                          stride=2,
                                          bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.act1 = ActFunc('PReLU', num_parameters=out_channels // 2)

        self.conv_block = _make_conv_layer(out_channels, n_convs)
        self.act2 = ActFunc('PReLU', num_parameters=out_channels)


    def forward(self, x, skipx):
        x = self.de_conv(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = torch.cat((x, skipx), dim=1)
        out = self.conv_block(x)

        out = out + x
        out = self.act2(out)

        return out


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.in_tr = InputTransition()
        self.down_tr32 = DownTransition(in_channels=16, n_convs=1)
        self.down_tr64 = DownTransition(in_channels=32, n_convs=2)
        self.down_tr128 = DownTransition(in_channels=64, n_convs=3)
        self.down_tr256 = DownTransition(in_channels=128, n_convs=3)
        self.up_tr256 = UpTransition(in_channels=256, out_channels=256, n_convs=2)
        self.up_tr128 = UpTransition(in_channels=256, out_channels=128, n_convs=2)
        self.up_tr64 = UpTransition(in_channels=128, out_channels=64, n_convs=1)
        self.up_tr32 = UpTransition(in_channels=64, out_channels=32, n_convs=1)
        self.out_tr = OutputTransition(in_channels=32)

        self.attention1 = AttentionGate(in_channels=16, gating_channels=256, inter_channels=8)
        self.attention2 = AttentionGate(in_channels=32, gating_channels=256, inter_channels=16)
        self.attention3 = AttentionGate(in_channels=64, gating_channels=256, inter_channels=32)
        self.attention4 = AttentionGate(in_channels=128, gating_channels=256, inter_channels=64)


    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        g_out16 = self.attention1(out16, out256)
        g_out32 = self.attention2(out32, out256)
        g_out64 = self.attention3(out64, out256)
        g_out128 = self.attention4(out128, out256)

        out = self.up_tr256(out256, g_out128)
        out = self.up_tr128(out, g_out64)
        out = self.up_tr64(out, g_out32)
        out = self.up_tr32(out, g_out16)
        out = self.out_tr(out)

        # Flatten result
        return out


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TestNet()

    model = model.to(device)

    from torchsummary import summary
    summary(model, input_size=(1, 64, 128, 128))
    print(model)