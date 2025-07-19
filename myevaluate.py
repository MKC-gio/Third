import sys
import numpy as np
import random
import cv2
import os
import torch
import yaml
import torchvision
import argparse
import scipy
import torchgeometry as tgm
import datasets_haze_voc as datasets
from omegaconf import DictConfig, OmegaConf
from utils import *
from glob import glob
from models.VFIformer_arch import VFIformerSmall
import os.path as osp
from tqdm import tqdm
from models.warplayer import warp
import time
import logging
from PIL import Image, ImageDraw

def evaluate_SNet(model, val_loader, cfgs, device):
    mace_list = []
    for i_batch, inputs in enumerate(val_loader):
        print(i_batch)
        img0 = inputs['img0'].cuda(1)
        img1 = inputs['img1'].cuda(1)
        # print(f"img0 shape after transpose: {img0.shape}")
        flow_gt = inputs['flow_gt'].cuda(1)
        # print(f"flow_gt1 shape: {flow_gt.shape}")

        flow_gt_src = inputs['flow_gt'].squeeze(0).detach().numpy()
        flow_gt_dst = inputs['flow_gt'].squeeze(0).detach().numpy()

        pre_flow, fea_32 = model.forward(img0, img1)

        # Compute Homography
        H, status = cv2.findHomography(flow_gt_dst.reshape(-1, 2), flow_gt_src.reshape(-1, 2))

        # Prepare ground truth for calculation
        four_point_org = torch.zeros((2, 2, 2)).cuda(1)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0]).cuda(1)
        four_point_org[:, 0, 1] = torch.Tensor([128 - 1, 0]).cuda(1)
        four_point_org[:, 1, 0] = torch.Tensor([0, 128 - 1]).cuda(1)
        four_point_org[:, 1, 1] = torch.Tensor([128 - 1, 128 - 1]).cuda(1)

        four_point = torch.zeros((2, 2, 2)).cuda(1)
        four_point[:, 0, 0] = flow_gt[:, 0, 0] + torch.Tensor([0, 0]).cuda(1)
        four_point[:, 0, 1] = flow_gt[:, 0, 1] + torch.Tensor([128 - 1, 0]).cuda(1)
        four_point[:, 1, 0] = flow_gt[:, 1, 0] + torch.Tensor([0, 128 - 1]).cuda(1)
        four_point[:, 1, 1] = flow_gt[:, 1, 1] + torch.Tensor([128 - 1, 128 - 1]).cuda(1)

        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0)
        four_point = four_point.flatten(1).permute(1, 0).unsqueeze(0)

        # Calculate perspective transform
        H = tgm.get_perspective_transform(four_point_org, four_point)
        H = H.squeeze()

        # Calculate flow points
        flow = pre_flow.squeeze(0)
        four_point1 = torch.zeros((2, 2, 2)).cuda(1)
        four_point1[:, 0, 0] = flow[:, 0, 0] + torch.Tensor([0, 0]).cuda(1)
        four_point1[:, 0, 1] = flow[:, 0, -1] + torch.Tensor([128 - 1, 0]).cuda(1)
        four_point1[:, 1, 0] = flow[:, -1, 0] + torch.Tensor([0, 128 - 1]).cuda(1)
        four_point1[:, 1, 1] = flow[:, -1, -1] + torch.Tensor([128 - 1, 128 - 1]).cuda(1)

        four_point1 = four_point1.flatten(1).permute(1, 0).unsqueeze(0)
        H1 = tgm.get_perspective_transform(four_point_org, four_point1)
        H1 = H1.squeeze()

        # Compute MACE (Mean Absolute Cosine Error)
        flow_4cor = torch.zeros((1, 2, 2, 2)).cuda(1)
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

        mace = torch.sum((pre_flow[0, :, :, :].to(device) - flow_4cor) ** 2, dim=0).sqrt()
        # mace = torch.sum((pre_flow[0, :, :, :].cpu() - flow_4cor) ** 2, dim=0).sqrt()
        mace_list.append(mace.mean().item())  # Store the MACE for later averaging

    # Calculate average MACE
    mace = np.mean(mace_list)
    print("Validation MACE: %f" % mace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-config', type=str, default='/media/mygo/partition2/zzx/shizeru/Meta-Homo/config/test.yaml',
                        help='Path to the configuration (YAML format)')
    
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
    device = torch.device('cuda:'+ str(cfgs.gpuid))

    model = VFIformerSmall(cfgs)
    # 3
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_f3_rain/81_pretrain_net_20.pth',map_location='cpu')
    # 4
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_f4_rain/81_pretrain_net_30.pth',map_location='cpu')
    # 5
    model_med = torch.load('/media/mygo/partition2/zzx/shizeru/CKM/meta2/train_haze1/MetaHomo_0round_20epoch.pth',map_location='cpu')
    # 6
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_f6_rain/81_pretrain_net_30.pth',map_location='cpu')
    # 7
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_f8_rain/81_pretrain_net_20.pth',map_location='cpu')
    model.load_state_dict(model_med['net'])
    model.to(device) 
    model.eval()

    val_dataset = datasets.fetch_dataloader(cfgs)
    val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=cfgs.batch_size,
                shuffle=False,
                pin_memory=True,
            )
    evaluate_SNet(model, val_loader, cfgs, device)