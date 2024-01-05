import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet
from lib.utils.tools import read_pkl
from pyskl.datasets import build_dataset, build_dataloader
from vanilla.model.transformer import Transformer

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def permute_single_batch(batch):
    """
    Permute and reshape the given batch.
    
    Assumes the batch is a dictionary containing a tensor where
    the last two dimensions need to be combined.
    """
    batch['imgs'] = batch['imgs'].permute(0, 1, 3, 2, 4, 5)  # Permute the dimensions
    
    # Get the shape of the tensor and combine the last two dimensions
    *leading_dims, last_dim1, last_dim2 = batch['imgs'].shape
    batch['imgs'] = batch['imgs'].reshape(*leading_dims, last_dim1*last_dim2)
    batch['imgs'] = batch['imgs'].permute(0, 4, 2, 3, 1)
    
    batch['label'] = batch['label'].squeeze(dim=1)

    return batch

def permute_test_data(batch):
    """
    Permute and reshape the given batch.
    
    Assumes the batch is a dictionary containing a tensor where
    the last two dimensions need to be combined.
    """
    batch['imgs'] = batch['imgs'].permute(0, 1, 3, 2, 4, 5)  # Permute the dimensions
    
    # Get the shape of the tensor and combine the last two dimensions
    *leading_dims, last_dim1, last_dim2 = batch['imgs'].shape
    batch['imgs'] = batch['imgs'].reshape(*leading_dims, last_dim1*last_dim2)
    batch['imgs'] = batch['imgs'].permute(0, 4, 2, 3, 1)

    return batch

dataset = read_pkl('data/action/ntu60_hrnet.pkl')
dataset_type = 'PoseDataset'
ann_file = 'data/action/ntu60_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=False, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=32),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', pipeline=train_pipeline)),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=test_pipeline))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    opts = parser.parse_args()
    return opts

def validate(test_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(test_loader[0]):
            batch_data = permute_test_data(batch_data)
            batch_input = batch_data['imgs']
            batch_gt = batch_data['label']
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+8) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
    print('TEST RESULTS:')
    print('Test loss: ', losses.avg, 'Test top1: ', top1.avg, 'Test top5: ', top5.avg)
    return losses.avg, top1.avg, top5.avg


def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = Transformer(dim_in=3136, num_classes=60, dim_feat=768,
                        depth=12, num_heads=12, mlp_ratio=4,
                        num_frames=48, num_joints=17, patch_size=1, t_patch_size=4,
                        qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        drop_path_rate=0., norm_layer=nn.LayerNorm, protocol='linprobe')
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda() 
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')

    train_dataset = [build_dataset(data['train'])]
    test_dataset = [build_dataset(data['test'], dict(test_mode=True))]

    train_dataloader_setting = dict(
        videos_per_gpu=data.get('videos_per_gpu', 1),
        workers_per_gpu=data.get('workers_per_gpu', 1),
        persistent_workers=data.get('persistent_workers', False))
    train_dataloader_setting = dict(train_dataloader_setting,
                              **data.get('train_dataloader', {}))
    
    test_dataloader_setting = dict(
            videos_per_gpu=data.get('videos_per_gpu', 1),
            workers_per_gpu=data.get('workers_per_gpu', 1),
            persistent_workers=data.get('persistent_workers', False),
            shuffle=False)
    test_dataloader_setting = dict(test_dataloader_setting,
                                **data.get('test_dataloader', {}))

    train_data_loaders = [
        build_dataloader(ds, **train_dataloader_setting) for ds in train_dataset
    ]

    test_data_loaders = [
        build_dataloader(ds, **test_dataloader_setting) for ds in test_dataset
    ]

   
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr_backbone,
            weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_data_loaders[0])))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_data_loaders)
            for idx, batch_data in enumerate(train_data_loaders[0]):    # (N, 2, T, 17, 3) to (N, 1, T, 17, 3136)
                batch_data = permute_single_batch(batch_data)
                batch_input = batch_data['imgs']
                batch_gt = batch_data['label']
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                output = model(batch_input) # (N, num_classes)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()    
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_data_loaders), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1))
                sys.stdout.flush()
                
            test_loss, test_top1, test_top5 = validate(test_data_loaders, model, criterion)
                
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('test_top5', test_top5, epoch + 1)
            
            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if test_top1 > best_acc:
                best_acc = test_top1
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

    if opts.evaluate:
        test_loss, test_top1, test_top5 = validate(test_data_loaders, model, criterion)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)
