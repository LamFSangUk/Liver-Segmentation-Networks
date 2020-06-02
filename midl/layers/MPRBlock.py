import torch
import torch.nn as nn

from midl.layers.ActFunc import ActFunc

class MPRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MPRBlock, self).__init__()
        assert out_channels % 4 == 0

        inter_channels = out_channels//4

        self.axial = nn.Conv3d(in_channels=in_channels,
                               out_channels=inter_channels,
                               kernel_size=(1, 3, 3),
                               stride=(1,1,1),
                               padding=(0,1,1),
                               bias=False)
        self.bn_axial = nn.BatchNorm3d(inter_channels)
        self.act_axial = ActFunc('ReLU')

        self.coronal = nn.Conv3d(in_channels=in_channels,
                                 out_channels=inter_channels,
                                 kernel_size=(3, 1, 3),
                                 stride=(1,1,1),
                                 padding=(1,0,1),
                                 bias=False)
        self.bn_coronal = nn.BatchNorm3d(inter_channels)
        self.act_coronal = ActFunc('ReLU')

        self.sagittal = nn.Conv3d(in_channels=in_channels,
                                 out_channels=inter_channels,
                                 kernel_size=(3, 3, 1),
                                 stride=(1, 1, 1),
                                 padding=(1, 1, 0),
                                 bias=False)
        self.bn_sagittal = nn.BatchNorm3d(inter_channels)
        self.act_sagittal = ActFunc('ReLU')

        self.all = nn.Conv3d(in_channels=in_channels,
                                 out_channels=inter_channels,
                                 kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=False)
        self.bn_all = nn.BatchNorm3d(inter_channels)
        self.act_all = ActFunc('ReLU')

    def forward(self, x):
        out_axial = self.act_axial(self.bn_axial(self.axial(x)))
        out_coronal = self.act_coronal(self.bn_coronal(self.coronal(x)))
        out_sagittal = self.act_sagittal(self.bn_sagittal(self.sagittal(x)))
        out_total = self.act_all(self.bn_all(self.all(x)))

        out = torch.cat([out_axial, out_coronal, out_sagittal, out_total], dim=1)
        out = out + x       # Residual

        return out
