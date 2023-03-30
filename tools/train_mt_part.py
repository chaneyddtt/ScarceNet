
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate_mt, AverageMeter, validate
from core.function1 import train_mt_update
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.augmentation_pool import RandAugmentMC
from utils.consistency import prediction_check
from core.evaluate import accuracy
import models
import dataset_animal


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--animalpose',
                        help='train on ap10k',
                        action='store_true')

    parser.add_argument('--fewshot',
                        help='train on ap10k with few shot annotations',
                        action='store_true')

    parser.add_argument('--num_transforms',
                        help='number of transformations used for generating pseudo labels',
                        type=int,
                        default=5)
    parser.add_argument('--generate_pseudol',
                        help='set true generate pseudo labels',
                        action='store_true')

    parser.add_argument('--pretrained',
                        help='path for pretrained model',
                        type=str,
                        default='')
    parser.add_argument('--resume', help='path to resume', type=str, default='')

    parser.add_argument('--const_weight', type=float, default=2.0)
    parser.add_argument('--consistency_rampup', type=int, default=10)

    parser.add_argument('--dist_thre', type=float, default=0.6)
    parser.add_argument('--update_epoch', type=int, default=-1)

    parser.add_argument('--topkrampdown', type=int, default=30)
    parser.add_argument('--minrate', type=float, default=0.8)

    parser.add_argument('--score_thre', type=float, default=0.5)

    parser.add_argument('--sf_aggre', type=float, default=0.1)
    parser.add_argument('--rf_aggre', type=float, default=20)
    parser.add_argument('--length', type=int, default=32)
    parser.add_argument('--nholes', type=int, default=6)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--std_gaussian', type=float, default=0.2)

    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--true_label_w', type=float, default=2.0)

    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--schedule', type=int, nargs='+', default=[190, 200],
                        help='Decrease learning rate at these epochs.')

    parser.add_argument('--percentage', type=float, default=0.4)

    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--n', type=int, default=2)

    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--few_shot_setting', action='store_false')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    model_ema = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    for param in model_ema.parameters():
        param.detach()

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/core/function1.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    logger.info(get_model_summary(model, dump_input))

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if args.animalpose:
        if args.augment:
            logger.info("Add strong augmentations to student")
            transfm_stu = transforms.Compose([RandAugmentMC(n=args.n, m=args.m, num_cutout=args.nholes),
                                              transforms.ToTensor(),
                                              normalize])
            transfm_tea = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            logger.info("Without strong augmentations to student")
            transfm_stu = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ])

            transfm_tea = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ])

        train_dataset = eval('dataset_animal.' + 'ap10k')(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_dataset = eval('dataset_animal.' + 'ap10k')(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    last_epoch = -1
    best_perf_epoch = 0
    optimizer = get_optimizer(cfg, model)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    # load pretrained model
    if os.path.exists(args.pretrained):
        logger.info("=> loading checkpoint '{}'".format(args.pretrained))
        pretrained_model = torch.load(args.pretrained)
        model.load_state_dict(pretrained_model['state_dict'])

    # resume pretrained model
    if os.path.exists(args.resume):
        logger.info("=> resume from checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model_ema = torch.nn.DataParallel(model_ema, device_ids=cfg.GPUS).cuda()


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.schedule, args.lr_factor,
        last_epoch=last_epoch
    )

    if args.evaluate:
        acc = validate(cfg, valid_loader, valid_dataset, model, criterion,
                        final_output_dir, tb_log_dir, writer_dict, args.animalpose)
        return

    for epoch in range(begin_epoch, args.epochs):
        if epoch == begin_epoch:
            if args.generate_pseudol:
                model.eval()
                train_dataset = eval('dataset_animal.ap10k')(
                    cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, False,
                    # transforms.Compose([
                    #     transforms.ToTensor(),
                    #     normalize,
                    # ])
                )
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
                    shuffle=False,
                    num_workers=cfg.WORKERS,
                    pin_memory=cfg.PIN_MEMORY
                )
                pseudo_kpts = {}
                acc_pseudol = AverageMeter()
                for _, (input, target, target_weight, meta) in enumerate(train_loader):
                    for i in range(input.size(0)):
                        c = meta['center'].numpy()
                        s = meta['scale'].numpy()
                        generated_kpts, score_map = prediction_check(input[i], model, train_dataset, c[i:i+1], s[i:i+1],
                                                                     args.num_transforms)
                        pseudo_kpts[int(meta['index'][i].numpy())] = generated_kpts
                        _, avg_acc, cnt, pred = accuracy(score_map,
                                                         target[i].unsqueeze(0).numpy())
                        acc_pseudol.update(avg_acc, cnt)
                print("Acc on the training dataset (pseudo-labels): {}".format(acc_pseudol.avg))
                np.save('data/pseudo_labels/{}shots/pseudo_labels_train.npy'.format(cfg.LABEL_PER_CLASS),
                            pseudo_kpts)
                break

            train_dataset = eval('dataset_animal.' + cfg.DATASET.DATASET)(
                cfg, args, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                transfm_stu, transfm_tea
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
                shuffle=cfg.TRAIN.SHUFFLE,
                num_workers=cfg.WORKERS,
                pin_memory=cfg.PIN_MEMORY,
                drop_last=True
            )

        lr_scheduler.step()
        train_mt_update(cfg, args, train_loader, train_dataset, model, model_ema, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        perf_indicator = validate_mt(
            cfg, valid_loader, valid_dataset, model, model_ema, criterion,
            final_output_dir, tb_log_dir, writer_dict, args.animalpose)

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
            best_perf_epoch = epoch + 1
        else:
            best_model = False
        # save model
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model_ema.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    logger.info('Best accuracy {} at epoch {}'.format(best_perf, best_perf_epoch))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
