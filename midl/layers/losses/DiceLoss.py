import torch
import torch.nn as nn

import numpy as np

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

"""
https://github.com/mattmacy/vnet.pytorch/blob/master/train.py
"""

# class DiceLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, target):
#         ctx.save_for_backward(input, target)
#
#         eps = 0.0001
#         _, result_ = input.max(1)
#         result_ = torch.squeeze(result_)
#         if input.is_cuda:
#             result = torch.cuda.FloatTensor(result_.size())
#             ctx.target_ = torch.cuda.FloatTensor(target.size())
#         else:
#             result = torch.FloatTensor(result_.size())
#             ctx.target_ = torch.FloatTensor(target.size())
#         result.copy_(result_)
#         ctx.target_.copy_(target)
#         target = ctx.target_
# #       print(input)
#         intersect = torch.dot(result, target)
#         # binary values so sum the same as sum of squares
#         result_sum = torch.sum(result)
#         target_sum = torch.sum(target)
#         union = result_sum + target_sum + (2*eps)
#
#         # the target volume can be empty - so we still want to
#         # end up with a score of 1 if the result is 0/0
#         IoU = intersect / union
#         # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#         #     union, intersect, target_sum, result_sum, 2*IoU))
#         out = torch.FloatTensor(1).fill_(2*IoU)
#         ctx.intersect, ctx.union = intersect, union
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, _ = ctx.saved_variables
#         intersect, union = ctx.intersect, ctx.union
#         target = ctx.target_
#         gt = torch.div(target, union)
#         IoU2 = intersect/(union*union)
#         pred = torch.mul(input[:, 1], IoU2)
#         dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
#         grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
#                                 torch.mul(dDice, grad_output[0])), dim=0)
#
#         grad_input = grad_input.contiguous().view(input.shape)
#
#         return grad_input , None



# import torch
# from torch import nn
#
# class DiceLoss(nn.Module):
#     """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
#     Useful in dealing with unbalanced data
#     Add softmax automatically
#
#     https://github.com/thisissum/dice_loss/blob/master/dice_loss.py
#     """
#
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, y_pred, y_true):
#         # shape(y_pred) = batch_size, label_num, **
#         # shape(y_true) = batch_size, **
#         pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))
#         dsc_i = 1 - ((1-pred_prob)*pred_prob) / ((1-pred_prob) * pred_prob + 1)
#         dice_loss = dsc_i.mean()
#
#         return dice_loss
"""https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py"""
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        numerator = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        denominator = torch.sum(torch.add(predict, target), dim=1) + self.smooth

        loss = 1 - numerator / denominator

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target, weight):
        assert(predict.shape == target.shape)
        dice = BinaryDiceLoss()

        total_loss = 0.0
        for i in range(target.shape[1]):
            dice_loss = dice(predict[:, i], target[:, i])
            dice_loss *= weight[i]

            total_loss += dice_loss

        return total_loss / target.shape[1]

