import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from skimage.metrics import structural_similarity
from utils import AverageMeter,chw_to_hwc
from datasets.loader import PairLoader, TPairLoader
from  datasets.loader import MixUp_AUG
from models import *
import cv2
import random
import numpy as np
from torchprofile import profile_macs
from skimage import color
from collections import OrderedDict
from ptcolor import rgb2lab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='spmformer-b', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='LSUI', type=str, help='dataset name')
parser.add_argument('--exp', default='lsui', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def fourier_loss(input_image, target_image):
    # input_freq = torch.zeros_like(input_image, dtype=torch.complex64)
    # target_freq = torch.zeros_like(input_image, dtype=torch.complex64)
    # Apply Fourier Transform to input and target images
    input_freq = torch.fft.fft2(input_image)
    input_abs = torch.abs(input_freq)  # 幅度谱，求模得到
    input_ang = torch.angle(input_freq)  # 相位谱，求相角得到
    target_freq = torch.fft.fft2(target_image)
    target_abs = torch.abs(target_freq)  # 幅度谱，求模得到
    target_ang = torch.angle(target_freq)  # 相位谱，求相角得到

    # Calculate squared magnitude difference in frequency domain
    #freq_diff = torch.abs(input_freq) - torch.abs(target_freq)
    abs_loss = criterion(input_abs, target_abs)
    ang_loss = criterion(input_ang, target_ang)

    return abs_loss + ang_loss

def train(train_loader, network, criterion, optimizer, scaler, epoch):
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        #mask_img = batch['mask'].cuda()

        # if epoch > 70:
        #    target_img, source_img, mask_img = MixUp_AUG().aug(target_img, source_img, mask_img)

        with autocast(args.no_autocast):
            output = network(source_img)  # [batch, c, h, w] -> c-[b,g,r]
            target_gr = torch.ones_like(output)  
            tones = torch.ones_like(output)
            #ones[:,2,:,:] *= 3
            target_non_zero_indices = target_img != 0  # 找到vector2中不为零的位置
            target_gr[target_non_zero_indices] = (output[target_non_zero_indices] / target_img[target_non_zero_indices])

            
            rgb_output = output[:, [2, 1, 0], :, :] * 0.5 + 0.5
            rgb_target = target_img[:, [2, 1, 0], :, :] * 0.5 + 0.5
            # lab_output = rgb2lab(rgb_output)
            # lab_target = rgb2lab(rgb_target)
            lab_output = torch.clamp(rgb2lab(rgb_output), -80.0, 80.0)
            lab_target = torch.clamp(rgb2lab(rgb_target), -80.0, 80.0)
            labones = torch.ones_like(lab_output)
            lab_ratio = torch.ones_like(lab_output)
            target_non_zero_indices1 = lab_target != 0  # 找到vector2中不为零的位置
            lab_ratio[target_non_zero_indices1] = (lab_output[target_non_zero_indices1] / lab_target[target_non_zero_indices1])

            loss = criterion(output, target_img) + 0.01 * criterion(tones, target_gr) + 0.01 * criterion(lab_output, lab_target) + 0.01 * fourier_loss(output, target_img)
        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        #mask_img = batch['mask'].cuda()

        with torch.no_grad():							# torch.no_grad() may cause warning
            output = network(source_img).clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))
    return PSNR.avg

def get_model_size(model: torch.nn.Module) -> float:
    num_params = sum(p.numel() for p in model.parameters())
    #num_bytes = num_params * 4  # assuming 32-bit float
    num_megabytes = num_params / (1000 ** 2)
    return num_megabytes

if __name__ == '__main__':
    calculate = True
    setting_filename = os.path.join('configs', args.exp, args.model+'.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()
    #network = nn.DataParallel(network, device_ids=[0]).cuda()

    #criterionL2 = nn.SmoothL1Loss()
    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    elif setting['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min= 1e-6)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)

    #UIEB dataset->800 / 90
    # dataset = PairLoader(dataset_dir, 'train', 'train',
    #                            setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 90])
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=setting['batch_size'],
    #                           shuffle=True,
    #                           num_workers=args.num_workers,
    #                           pin_memory=True,
    #                           drop_last=True)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=setting['batch_size'],
    #                         num_workers=args.num_workers,
    #                         pin_memory=True)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                                setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = TPairLoader(dataset_dir, 'test', setting['valid_mode'],
                              setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))
        
        calculate = True
        if calculate is True:
        # calculate the number of parameters
            model_size = get_model_size(network)
            print(f"Model size: {model_size:.2f} MB")
            calculate = False
        best_psnr = 0
        

        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler,epoch)
            print(loss)
            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                print(f'current psnr:{avg_psnr:.2f}\t, best psnr:{best_psnr:.2f}')
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))

                writer.add_scalar('best_psnr', best_psnr, epoch)
            #torch.save({'state_dict': network.state_dict()},
           # os.path.join(save_dir, args.model + 'final.pth'))

    else:
        print('==> Existing trained model')
        exit(1)

