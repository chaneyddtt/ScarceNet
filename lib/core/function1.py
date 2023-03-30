from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import torch
from core.evaluate import accuracy
from core.inference import get_final_preds, get_final_preds_const, get_max_preds
from utils.transforms import flip_back, fliplr_joints_batch, fliplr_weights_batch, fliplr_joints_batch_v2
from utils.utils import update_ema_variables, get_current_consistency_weight, get_current_topkrate
from core.function import AverageMeter
from core.loss import select_small_loss_samples_v2
logger = logging.getLogger(__name__)


def train_mt_update(config, args, train_loader, dataset, model, model_ema, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sup = AverageMeter()
    losses_consistency = AverageMeter()
    acc = AverageMeter()
    acc_ema = AverageMeter()
    # switch to train mode
    model.train()
    model_ema.train()
    end = time.time()

    # ratio of small loss samples, reduce gradually to avoid overfitting to initial pseudo labels
    topk_rate = get_current_topkrate(epoch, args.topkrampdown, args.minrate)
    for i, (input, target, target_weight, input_ema, input1, meta) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        input_ema = input_ema.cuda()
        input2 = input_ema.detach().clone()
        input1 = input1.cuda()
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        use_label = meta['use_label'].cuda(non_blocking=True)
        flip_var = meta['flip'].numpy()
        img_width = meta['img_width'].numpy()
        warpmat1 = meta['warpmat1'].numpy()
        warpmat2 = meta['warpmat_v2'].numpy()
        flip_var_v2 = meta['flip_v2'].numpy()
        # compute output
        outputs, _ = model(input)
        outputs_copy = outputs.detach().clone()

        # select small loss samples
        small_loss_idx = select_small_loss_samples_v2(outputs_copy, target, target_weight, topk_rate)
        weights_small_loss = torch.zeros_like(target_weight)
        weights_small_loss[small_loss_idx[:, 0], small_loss_idx[:, 1], 0] = 1

        if epoch > args.update_epoch:
            with torch.no_grad():
                outputs1, _ = model(input1)
                outputs2, _ = model(input2)
            # compute joint locations for input_v1, in the scale of heatmap size
            joints1, _ = get_max_preds(outputs1.detach().cpu().numpy())
            c2 = meta['center_ema'].numpy()
            s2 = meta['scale_ema'].numpy()

            joints2_vis = meta['joints_vis_ema'][:, :, :2].numpy()
            # compute the joint locations for input_v2, in the scale of original image size
            joints2, _ = get_final_preds_const(outputs2.cpu().numpy(), c2, s2)
            # flip output_v2 to keep in consistent with input_v1
            joints2_flip, joints2_vis_flip = fliplr_joints_batch_v2(joints2, joints2_vis, img_width[:, None],
                                                               dataset.flip_pairs)
            joints2_1 = np.where(flip_var_v2[:, None, None], joints2_flip, joints2)
            joints2_1 = np.concatenate([joints2_1[:, :, :2], torch.ones(joints2_1.shape[0], joints2_1.shape[1], 1)], axis=-1)
            # transform output_v2 to keep in consistent with input_v1
            joints21_trans = np.matmul(warpmat2, np.transpose(joints2_1, (0, 2, 1)))
            joints21_trans = np.transpose(joints21_trans, (0, 2, 1))
            joints21_trans_hmsize = joints21_trans/4 + 0.5
            # compute distance between input_v1 and input_v2
            dist = np.linalg.norm(joints1 - joints21_trans_hmsize, axis=-1, keepdims=True)

            # use the current model prediction as supervision, hence needs to transform the prediction to keep
            # consistent with the input
            joints2_ori = np.where(flip_var[:, None, None], joints2_flip, joints2)
            joints2_ori = np.concatenate([joints2_ori[:, :, :2], torch.ones(joints2_ori.shape[0], joints2_ori.shape[1], 1)], axis=-1)
            joints2_ori_trans = np.matmul(warpmat1, np.transpose(joints2_ori, (0, 2, 1)))
            joints2_ori_trans = np.transpose(joints2_ori_trans, (0, 2, 1))

            # agreement check to select reusable samples
            mask = (dist < args.dist_thre)
            weights2 = mask.astype(float)
            # exclude the small loss samples
            weights2 = weights2 * (1 - weights_small_loss.cpu().numpy())

            # re-labeling for reusable samples
            hms2_re = np.zeros_like(target.cpu().numpy())
            targets_weight2_re = np.zeros_like(target_weight.cpu().numpy())
            for b in range(hms2_re.shape[0]):
                hm2_re, target_weight2_re = dataset.generate_target(joints2_ori_trans[b], weights2[b])
                hms2_re[b] = hm2_re
                targets_weight2_re[b] = target_weight2_re
            hms2_re = torch.from_numpy(hms2_re).cuda()
            targets_weight2_re = torch.from_numpy(targets_weight2_re).cuda()

            mask_ = torch.tensor(weights2 > 0).cuda()
            # use model prediction for reusable samples, and the initial pseudo label otherwise
            target2 = torch.where(mask_[:, :, :, None], hms2_re, target)
            targets_weight2 = targets_weight2_re + weights_small_loss
            assert targets_weight2.max() < 2

            # increase the weights for supervised loss
            target = torch.where(use_label[:, None, None, None], target, target2)
            target_weight = args.true_label_w * target_weight
            target_weight = torch.where(use_label[:, None, None], target_weight, targets_weight2)

        if isinstance(outputs, list):
            loss_sup = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss_sup += criterion(output, target, target_weight)
        else:
            output = outputs
            loss_sup = criterion(output, target, target_weight)

        with torch.no_grad():
            if epoch > args.update_epoch:
                # feed the student input into the teacher network although we do not use it. This is to avoid
                # the batch norm statistics become too different for student and teacher
                _, _ = model_ema(input)
            outputs_ema, _ = model_ema(input_ema)

        # student-teacher consistency
        c_ema = meta['center_ema'].numpy()
        s_ema = meta['scale_ema'].numpy()
        joints_vis = meta['joints_vis_ema'][:, :, :2].numpy()
        joints_ori, _ = get_final_preds_const(outputs_ema.cpu().numpy(), c_ema, s_ema)
        # transform the teacher output to keep consistent with the student network
        joints_ori_flip, joints_vis_flip = fliplr_joints_batch(joints_ori, joints_vis, img_width[:, None], dataset.flip_pairs)
        joints = np.where(flip_var[:, None, None], joints_ori_flip, joints_ori)
        joints_vis = np.where(flip_var[:, None, None], joints_vis_flip, joints_vis)
        joints = np.concatenate([joints[:, :, :2], torch.ones(joints.shape[0], joints.shape[1], 1)], axis=-1)
        joints_trans = np.matmul(warpmat1, np.transpose(joints, (0, 2, 1)))
        joints_trans = np.transpose(joints_trans, (0, 2, 1))

        hms_ema_re = np.zeros_like(target.cpu().numpy())
        targets_weight_ema_re = np.zeros_like(target_weight.cpu().numpy())
        for b in range(hms_ema_re.shape[0]):
            hm_ema_re, target_weight_ema_re = dataset.generate_target(joints_trans[b], joints_vis[b])
            hms_ema_re[b] = hm_ema_re
            targets_weight_ema_re[b] = target_weight_ema_re

        hms_ema_re = torch.from_numpy(hms_ema_re).cuda()
        target_weight_ema_re = torch.from_numpy(targets_weight_ema_re).cuda()

        loss_consistency = criterion(output, hms_ema_re, target_weight_ema_re)
        const_loss_weight = get_current_consistency_weight(args.const_weight, epoch, args.consistency_rampup)
        loss = const_loss_weight * loss_consistency + loss_sup
        # compute gradient and do update step

        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, model_ema, 0.999, global_steps)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        losses_sup.update(loss_sup.item(), input.size(0))
        losses_consistency.update(loss_consistency, input.size(0))
        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        _, avg_acc_ema, cnt_ema, pred_ema = accuracy(outputs_ema.cpu().numpy(),
                                                     target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)
        acc_ema.update(avg_acc_ema, cnt_ema)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_sup {loss_sup.val:.5f} ({loss_sup.avg:.5f})\t' \
                  'Loss_const {loss_const.val:.5f} ({loss_const.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'\
                  'Accuracy_ema {acc_ema.val:.3f} ({acc_ema.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss_sup=losses_sup, loss_const=losses_consistency, acc=acc, acc_ema=acc_ema)
            logger.info(msg)

            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('loss_sup', losses_sup.val, global_steps)
            writer.add_scalar('loss_const', losses_consistency.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
