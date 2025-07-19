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

# logging.basicConfig(filename='/media/mygo/partition2/zzx/shizeru/ll_mace.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def evaluate_SNet(model, val_loader, cfgs, device):
    for i_batch, inputs in enumerate(val_loader):
        print(i_batch+1)
        img0 = inputs['img0'].cuda(1)
        img1 = inputs['img1'].cuda(1)
        # flow_gt = inputs['flow_gt'].cuda(1)
        
        pre_flow, fea_32= model.forward(img0,img1)
        flow = pre_flow.squeeze(0)
        
        four_point_org = torch.zeros((2, 2, 2)).cuda(1)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([128 - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, 128 - 1])
        four_point_org[:, 1, 1] = torch.Tensor([128 - 1, 128 - 1])
        
        four_point1 = torch.zeros((2, 2, 2)).cuda(1)
        four_point1[:, 0, 0] = flow[:, 0, 0] + torch.Tensor([0, 0]).cuda(1)
        four_point1[:, 0, 1] = flow[:, 0, -1] + torch.Tensor([128 - 1, 0]).cuda(1)
        four_point1[:, 1, 0] = flow[:, -1, 0] + torch.Tensor([0, 128 - 1]).cuda(1)
        four_point1[:, 1, 1] = flow[:, -1, -1] + torch.Tensor([128 - 1, 128 - 1]).cuda(1)
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0)
        four_point1 = four_point1.flatten(1).permute(1, 0).unsqueeze(0)
        H1 = tgm.get_perspective_transform(four_point_org,four_point1)
        H1 = H1.squeeze()
        H1 = H1.cpu().detach().numpy()
        
        
        img1 = img1.squeeze(0)
        img1 = img1.cpu().detach().permute(1,2,0).numpy()
        warp_img = cv2.warpPerspective(img1, H1, (128,128))
        
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask = cv2.warpPerspective(np.ones_like(mask) * 255, H1, (128, 128))
        
        image1 = img0.detach().cpu().numpy().squeeze(0)
        image1 = image1.transpose(1,2,0)
        image1 = cv2.resize(image1, (128, 128))

            
        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/VOC/haze/iters5/input1/{}.jpg'.format(i_batch+1),image1)
        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/VOC/haze/iters5/warp/{}.jpg'.format(i_batch+1),warp_img)
        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/VOC/haze/iters5/mask/{}.jpg'.format(i_batch+1),mask)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-config', type=str, default='/media/mygo/partition2/zzx/shizeru/Meta-Homo/config/test.yaml',
                        help='Path to the configuration (YAML format)')
    
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
    device = torch.device('cuda:'+ str(cfgs.gpuid))
    # setup_seed(2022)
    
    model = VFIformerSmall(cfgs)
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_f3_rain/81_pretrain_net_20.pth')
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_l4_ll/81_pretrain_net_20.pth')
    model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_l5_haze/81_pretrain_net_20.pth')
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_l6_ll/81_pretrain_net_20.pth')
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/CKM/meta2/pretrain_ll_iters7/81_pretrain_net_20.pth')
    
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_notran_haze/81_pretrain_net_20.pth')
    
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_rain_l5_truth/MetaHomo_0round_50epoch.pth')
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_ll_l5_truth/MetaHomo_0round_50epoch.pth')
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_haze_l5_truth/MetaHomo_0round_50epoch.pth')
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_rain_l5_truth/MetaHomo_0round_50epoch.pth')
    model.load_state_dict(model_med['net'])

    model.cuda(1) 
    model.eval()

    val_dataset = datasets.fetch_dataloader(cfgs)
    val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=cfgs.batch_size,
                shuffle=False,
                pin_memory=True,
            )
    evaluate_SNet(model, val_loader, cfgs, device)
    
