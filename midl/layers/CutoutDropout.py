import torch
import torch.nn as nn
import numpy as np


class CutoutDropout(nn.Module):
    def __init__(self, p, range_len_box=(1./6., 1./5.)):
        super(CutoutDropout, self).__init__()

        assert len(range_len_box) == 2

        if p < 0. or p > 1.:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        self.p = p
        self.min_range_len_box = range_len_box[0]
        self.max_range_len_box = range_len_box[1]

        self.random = torch.FloatTensor(1).uniform_

    def forward(self, x, training):
        out = x

        if training is True:
            if self.random().item() < self.p:
                depth, height, width = out.shape[2:]

                box_side = int(width*self.min_range_len_box)\
                           + int(self.random().item() * width * (self.max_range_len_box - self.min_range_len_box))
                box_depth = int(depth * self.min_range_len_box) \
                            + int(self.random().item() * depth * (self.max_range_len_box - self.min_range_len_box))

                z = torch.IntTensor([self.random().item() * depth]).cuda()
                y = torch.IntTensor([self.random().item() * height]).cuda()
                x = torch.IntTensor([self.random().item() * width]).cuda()

                z1 = torch.clamp(z - box_depth // 2, 0, depth)
                z2 = torch.clamp(z + box_depth // 2, 0, depth)
                y1 = torch.clamp(y - box_side // 2, 0, height)
                y2 = torch.clamp(y + box_side // 2, 0, height)
                x1 = torch.clamp(x - box_side // 2, 0, width)
                x2 = torch.clamp(x + box_side // 2, 0, width)

                mask = torch.ones((out.shape[2:])).cuda()
                mask[z1:z2, y1:y2, x1:x2] = 0

                out = out * mask



        return out