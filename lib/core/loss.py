# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


criterion_mse = nn.MSELoss(reduction='none')


def select_small_loss_samples_v2(output, target, target_weight, topk_rate):
    batch_size = output.size(0)
    number_joints = output.size(1)
    num_visible_joints = torch.count_nonzero(target_weight)
    num_small_loss_samples = int(num_visible_joints * topk_rate)
    output_re = output.reshape(batch_size, number_joints, -1)
    target_re = target.reshape(batch_size, number_joints, -1)
    loss = criterion_mse(output_re.mul(target_weight), target_re.mul(target_weight)).mean(-1)
    loss_max = loss.max() * torch.ones_like(loss)
    weight = (target_weight > 0)
    # set loss for joint with weight 0 to a large number to avoid being selected
    loss = torch.where(weight.squeeze(-1), loss, loss_max)
    dim_last = loss.size(-1)
    _, topk_idx = torch.topk(loss.flatten(), k=num_small_loss_samples, largest=False)
    topk_idx = topk_idx.unsqueeze(-1)
    idx_re = torch.cat([topk_idx // dim_last, topk_idx % dim_last], dim=-1)
    return idx_re


class CurriculumLoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(CurriculumLoss, self).__init__()
        self.criterion = nn.MSELoss(reduce=False)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, top_k):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))

        if self.use_target_weight:
            loss = 0.5 * (self.criterion(
                heatmaps_pred.mul(target_weight),
                heatmaps_gt.mul(target_weight)
            )).mean(-1)
        else:
            loss = 0.5 * (self.criterion(heatmaps_pred, heatmaps_gt)).mean(-1)
        weights_bool = (target_weight > 0)
        loss_clone = loss.clone().detach().requires_grad_(False)
        loss_inf = 1e8 * torch.ones_like(loss_clone, requires_grad=False)
        # set the loss of invalid joints (weights equal 0) to a large value such that it won't be
        # selected as reliable pseudo labels, only joints with smaller loss will be selected
        loss_clone = torch.where(weights_bool.squeeze(-1), loss_clone, loss_inf)
        _, topk_idx = torch.topk(loss_clone, k=top_k, dim=-1, largest=False)
        tmp_loss = torch.gather(loss, dim=-1, index=topk_idx)
        tmp_loss = tmp_loss.sum()/(top_k * batch_size)
        return tmp_loss