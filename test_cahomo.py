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
import torch.nn.functional as F

def evaluate_SNet(model, val_loader, cfgs, device):
    mace_list = []
    pme_list = []
    for i_batch, inputs in enumerate(val_loader):
        print(i_batch)
        img0 = inputs['img0'].cuda(1)
        img1 = inputs['img1'].cuda(1)
        flow_gt_src = np.array(inputs['flow_gt1'].squeeze(0).cpu().numpy(), dtype=np.float32)
        flow_gt_dst = np.array(inputs['flow_gt2'].squeeze(0).cpu().numpy(), dtype=np.float32)
        


        # # 获取语义特征
        # gd = F.interpolate(img1, size=(300, 300), mode='bilinear', align_corners=False)
        # fei = model.ssd_net.detect_feature(gd, 1)  # 使用SSD网络提取特征

        # 修改前向传播，加入语义特征
        pre_flow, fuj = model.forward(img0, img1)
        H, mask = cv2.findHomography(flow_gt_src, flow_gt_dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        H = H.squeeze()

        # 只保留内点
        flow_gt_src_filtered = flow_gt_src[mask.ravel() == 1]
        flow_gt_dst_filtered = flow_gt_dst[mask.ravel() == 1]
        
        flow = pre_flow.squeeze(0)
        B = flow.shape[0]

        # 构造四角坐标 ground truth
        four_point_org = torch.tensor([
            [0.0, 0.0],
            [127.0, 0.0],
            [0.0, 127.0],
            [127.0, 127.0]
        ], device=flow.device).unsqueeze(0).repeat(B, 1, 1)  # shape: [B, 4, 2]

        # 构造预测角点位置（基于 flow）
        four_point1 = torch.stack([
            flow[:, :, 0, 0] + torch.tensor([0, 0], device=flow.device),
            flow[:, :, 0, -1] + torch.tensor([127, 0], device=flow.device),
            flow[:, :, -1, 0] + torch.tensor([0, 127], device=flow.device),
            flow[:, :, -1, -1] + torch.tensor([127, 127], device=flow.device),
        ], dim=1)  # shape: [B, 4, 2]

        # 直接调用变换
        H1 = tgm.get_perspective_transform(four_point_org, four_point1)  # [B, 3, 3]
        H1 = H1.squeeze()

        # Robust PME computation for each batch element
        for b in range(B):
            H1_b = H1[b]  # [3, 3]

            # Prepare homogeneous coordinates
            points1_homogeneous = np.hstack([
                flow_gt_dst_filtered, 
                np.ones((flow_gt_dst_filtered.shape[0], 1), dtype=np.float32)
            ])  # [N, 3]

            points1_homogeneous = torch.tensor(points1_homogeneous, dtype=torch.float32).to(device)  # [N, 3]
            
            # Apply perspective transform
            points1_transformed_homogeneous = (H1_b @ points1_homogeneous.T).T  # [N, 3]
            points1_transformed = points1_transformed_homogeneous[:, :2] / points1_transformed_homogeneous[:, 2:3]  # [N, 2]

            # Ground truth source points
            flow_gt_src_filtered_tensor = torch.tensor(flow_gt_src_filtered, dtype=torch.float32).to(device)  # [N, 2]

            # Compute PME
            errors = torch.norm(flow_gt_src_filtered_tensor - points1_transformed, dim=1)
            PME = errors.mean().item()
            pme_list.append(PME)

    # 计算最终 PME 指标
    PME = np.mean(pme_list) if len(pme_list) > 0 else float("inf")
    print(f"Validation PME: {PME:.6f}")
    return PME



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='/media/mygo/partition2/zzx/shizeru/CKM/meta2/config/test.yaml',
                        help='Path to the configuration (YAML format)')
    
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
    device = torch.device('cuda:'+ str(cfgs.gpuid))

    # # 加载SSD模型
    # if cfgs.name == 'voc':
    #     cfg = voc
    # ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    # ssd_net.load_weights('/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_haze_l5_truth/pre_ssd.pth')
    # ssd_net.to(device)
    # ssd_net.eval()

    # 加载fusion模型
    model = VFIformerSmall(cfgs)
    model_med = torch.load('/media/mygo/partition2/zzx/shizeru/CKM/meta2/train_haze1/MetaHomo_0round_20epoch.pth',map_location='cpu')
    model.load_state_dict(model_med['net'])
    model.to(device) 
    model.eval()
    
    # # 将SSD模型添加到fusion模型中
    # model.ssd_net = ssd_net

    val_dataset = datasets.fetch_dataloader(cfgs)
    val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=cfgs.batch_size,
                shuffle=False,
                num_workers=cfgs.num_works,
                pin_memory=True,
            )
    evaluate_SNet(model, val_loader, cfgs, device)