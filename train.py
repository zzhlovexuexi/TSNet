import os
import argparse
import json

import numpy as np
import scipy.signal
from scipy.signal import wiener
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
from utils.CR_res import ContrastLoss_res

###
# from utils.CR_res import ContrastLoss_res1,ContrastLoss_res2
###


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='TSNet-t', type=str, help='model name')
parser.add_argument('--num_workers', default=12, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument(
    '--data_dir', default='/data', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network,network2, criterion,criterion2,optimizer,optimizer2, scaler,scaler2):
    losses = AverageMeter()
    losses2 = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        wu = target_img - source_img

        with autocast(args.no_autocast):
            output = network(source_img)

            qingxi = (source_img + output)

            loss = criterion[0](output, wu) + criterion[1](qingxi, target_img, source_img) * 0.1

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #22222222222222222222222222222222
        network2.train()
        qingxi=network2(qingxi.detach())

        loss2 = criterion2[0](qingxi, target_img) #+ criterion2[1](qingxi2, target_img, qingxi) * 0.1
        losses2.update(loss2.item())

        optimizer2.zero_grad()
        scaler2.scale(loss2).backward()
        scaler2.step(optimizer2)
        scaler2.update()

    return losses.avg,losses2.avg


def valid(val_loader, network,network2):
    PSNR1 = AverageMeter()
    PSNR2 = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()
    network2.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img)
            qingxi = (source_img + output)
            output=network2(qingxi).clamp_(-1, 1)

        mse_loss1 = F.mse_loss(qingxi * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr1 = 10 * torch.log10(1 / mse_loss1).mean()
        PSNR1.update(psnr1.item(), source_img.size(0))

        mse_loss2 = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr2 = 10 * torch.log10(1 / mse_loss2).mean()
        PSNR2.update(psnr2.item(), source_img.size(0))


    return PSNR1.avg,PSNR2.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # pretrain weights loader
    checkpoint = None
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()
    network2 = eval('TSNet_t2')()
    network2 = nn.DataParallel(network2).cuda()
    if checkpoint is not None:
        network.load_state_dict(checkpoint['state_dict'])

    criterion = []
    criterion2 = []
    criterion.append(nn.SmoothL1Loss(reduction='mean'))
    criterion.append(ContrastLoss_res(ablation=False).cuda())
    criterion2.append(nn.SmoothL1Loss(reduction='mean'))


    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
        optimizer2 = torch.optim.Adam(network2.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
        optimizer2 = torch.optim.AdamW(network2.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)

    scaler = GradScaler()
    scaler2 = GradScaler()

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        best_psnr = checkpoint['best_psnr']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_psnr1 = 0
        best_psnr2 = 0
        start_epoch = 0

    best_psnr1 = 0
    best_psnr2 = 0

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'],
                               setting['edge_decay'],
                               setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
    print('==> Start training, current model name: ' + args.model + ' + TSNet_t2')
    # print(network)


    train_ls, test_ls, idx = [], [], []
    train_ls2, test_ls2, idx2 = [], [], []

    for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):
        loss,loss2 = train(train_loader, network, network2, criterion, criterion2, optimizer, optimizer2, scaler, scaler2)

        train_ls.append(loss)
        idx.append(epoch)
        train_ls2.append(loss2)
        idx2.append(epoch)


        scheduler.step()
        scheduler2.step()

        if epoch % setting['eval_freq'] == 0:
            avg_psnr1,avg_psnr2 = valid(val_loader, network,network2)

            if avg_psnr1 > best_psnr1:
                best_psnr1 = avg_psnr1

            if avg_psnr2 > best_psnr2:
                best_psnr2 = avg_psnr2
                
                torch.save({'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            'best_psnr': best_psnr2
                            },
                           os.path.join(save_dir, args.model +'.pth'))

                torch.save({'state_dict': network2.state_dict(),
                            'optimizer': optimizer2.state_dict(),
                            'lr_scheduler': scheduler2.state_dict(),
                            'scaler': scaler2.state_dict(),
                            'epoch': epoch,
                            'best_psnr': best_psnr2
                            },
                           os.path.join(save_dir, args.model + '+ mix2' + '.pth'))
                
            print('loss:',loss,'/','loss2:',loss2,'--','best_psnr1:',best_psnr1,'/','best_psnr2:',best_psnr2)





