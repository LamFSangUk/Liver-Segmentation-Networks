import torch
import torch.nn as nn

from midl.layers.ActFunc import ActFunc

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.wx = nn.Conv3d(in_channels=in_channels,
                            out_channels=inter_channels,
                            kernel_size=1,
                            stride=1,
                            bias=False)
        self.wg = nn.Conv3d(in_channels=gating_channels,
                            out_channels=inter_channels,
                            kernel_size=1,
                            stride=1,
                            bias=False)

        self.psi = nn.Conv3d(in_channels=inter_channels,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             bias=False)
        self.act1 = ActFunc('ReLU')
        self.act2 = nn.Sigmoid()

        self.wout = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels)
        )

    def forward(self, x, g):
        wx = self.wx(x)
        wg = self.wg(nn.Upsample(size=wx.size()[2:], mode='trilinear')(g))

        psi = self.psi(self.act1(wx + wg))

        out = self.act2(psi)

        out = nn.Upsample(size=x.size()[2:], mode='trilinear')(out)
        out = out.expand_as(x) * x

        out = self.wout(out)

        return out
