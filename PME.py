import sys
import numpy as np
import torch
import cv2
import os
import yaml
import argparse
import datasets_haze_voc as datasets
from omegaconf import DictConfig, OmegaConf
from utils import *
from models.VFIformer_arch import VFIformerSmall
import os.path as osp
from tqdm import tqdm
import torchgeometry as tgm
import torch.nn.functional as F
from sklearn.linear_model import RANSACRegressor

def evaluate_SNet(model, val_loader, cfgs, device):
    pme_list = []
    
    for i_batch, inputs in enumerate(val_loader):
        print(i_batch)
        # print(f"img0 shape before transpose: {inputs['img0'].shape}")
        img0 = inputs['img0'].cuda(0)
        img1 = inputs['img1'].cuda(0)
        # print(f"img0 shape after transpose: {img0.shape}")
        
        # flow_gt = np.array(inputs['flow_gt1'].squeeze(0).cpu().numpy(), dtype=np.float32)
        
        flow_gt_src = np.array(inputs['flow_gt1'].squeeze(0).cpu().numpy(), dtype=np.float32)
        flow_gt_dst = np.array(inputs['flow_gt2'].squeeze(0).cpu().numpy(), dtype=np.float32)
        
        # print(f"flow_gt1 shape: {flow_gt.shape}")
        # print(f"flow_gt1 shape: {flow_gt_dst.shape}")
        
        pre_flow, _ = model.forward(img0, img1)
        # print(f"pre_flow shape: {pre_flow.shape}")
        
        # Compute Homography
        H, mask = cv2.findHomography(flow_gt_src, flow_gt_dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        H = H.squeeze()
        
        # 只保留内点
        flow_gt_src_filtered = flow_gt_src[mask.ravel() == 1]
        flow_gt_dst_filtered = flow_gt_dst[mask.ravel() == 1]
        
        # Prepare ground truth for calculation
        four_point_org = torch.zeros((2, 2, 2)).cuda(0)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0]).cuda(0)
        four_point_org[:, 0, 1] = torch.Tensor([128 - 1, 0]).cuda(0)
        four_point_org[:, 1, 0] = torch.Tensor([0, 128 - 1]).cuda(0)
        four_point_org[:, 1, 1] = torch.Tensor([128 - 1, 128 - 1]).cuda(0)
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0)
        
        
        # Calculate flow points
        flow = pre_flow.squeeze(0)
        four_point1 = torch.zeros((2, 2, 2)).cuda(0)
        four_point1[:, 0, 0] = flow[:, 0, 0] + torch.Tensor([0, 0]).cuda(0)
        four_point1[:, 0, 1] = flow[:, 0, -1] + torch.Tensor([128 - 1, 0]).cuda(0)
        four_point1[:, 1, 0] = flow[:, -1, 0] + torch.Tensor([0, 128 - 1]).cuda(0)
        four_point1[:, 1, 1] = flow[:, -1, -1] + torch.Tensor([128 - 1, 128 - 1]).cuda(0)
        

        four_point1 = four_point1.flatten(1).permute(1, 0).unsqueeze(0)
        # print(f"four_point_org shape: {four_point_org.shape}")
        # print(f"four_point1 shape: {four_point1.shape}")
        H1 = tgm.get_perspective_transform(four_point_org, four_point1)
        H1 = H1.squeeze()
        # H1 = H1.cpu().detach().numpy()
        # print("Homography Matrix H:")
        # print(H1)
        # img0 = img0.squeeze(0)
        # img0 = img0.cpu().detach().permute(1,2,0).numpy()
        # img1 = img1.squeeze(0)
        # img1 = img1.cpu().detach().permute(1,2,0).numpy()
        # warp_img = cv2.warpPerspective(img0, H, (128,128))
        # stitch = (img1+warp_img)/2
        # cv2.imwrite('/media/mygo/partition2/zzx/shizeru/CKM/other/results/e/{}.jpg'.format(i_batch+1), img0)
        # cv2.imwrite('/media/mygo/partition2/zzx/shizeru/CKM/other/results/ee/{}.jpg'.format(i_batch+1), img1)
        # cv2.imwrite('/media/mygo/partition2/zzx/shizeru/CKM/other/results/eee/{}.jpg'.format(i_batch+1), stitch)

        img1 = img1.squeeze(0)
        img1 = img1.cpu().detach().permute(1,2,0).numpy()
        img0 = img0.squeeze(0)
        img0 = img0.cpu().detach().permute(1,2,0).numpy()
        # 将 PyTorch 张量转换为 numpy 数组
        H1_np = H1.cpu().detach().numpy()
        warp_img = cv2.warpPerspective(img1, H1_np, (128,128))
        stitch = (img1+warp_img)/2
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask = cv2.warpPerspective(np.ones_like(mask) * 255, H1_np, (128, 128))
        cat = np.concatenate((img1, warp_img, stitch), axis=1)

        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/scale_matrix/8_16/CAhomo/rain/input1/{}.jpg'.format(i_batch+1),img1)
        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/scale_matrix/8_16/CAhomo/rain/warp/{}.jpg'.format(i_batch+1),warp_img)
        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/scale_matrix/8_16/CAhomo/rain/cat/{}.jpg'.format(i_batch+1),cat)
        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/scale_matrix/8_16/CAhomo/rain/stitch/{}.jpg'.format(i_batch+1),stitch)
        cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/scale_matrix/8_16/CAhomo/rain/mask/{}.jpg'.format(i_batch+1),mask)
        
        points1_homogeneous = np.hstack([flow_gt_dst_filtered, np.ones((flow_gt_dst_filtered.shape[0], 1), dtype='float32')])
        

        points1_homogeneous = torch.tensor(points1_homogeneous, dtype=torch.float32).cuda(device)
        
        # 应用单应性矩阵进行透视变换
        points1_transformed_homogeneous = (H1 @ points1_homogeneous.T).T

        # 归一化得到变换后的点
        points1_transformed = points1_transformed_homogeneous[:, :2] / points1_transformed_homogeneous[:, 2:3]

        
        # 计算欧几里得距离（平方误差）
        errors = np.linalg.norm(flow_gt_src_filtered - points1_transformed.cpu().detach().numpy(), axis=1)
        
        PME = np.mean(errors)
        # print(f"Validation PME: {PME:.6f}")
        # 记录当前 batch 的 PME 误差                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        pme_list.append(PME)

    # 计算最终 PME 指标
    PME = np.mean(pme_list) if len(pme_list) > 0 else float("inf")
    print(f"Validation PME: {PME:.6f}")
    return PME


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-config', type=str, default='/media/mygo/partition2/zzx/shizeru/Meta-Homo/config/test.yaml',
                        help='Path to the configuration (YAML format)')
    
    args = parser.parse_args()

    # Load config
    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))

    device = torch.device('cuda:'+ str(cfgs.gpuid))
    
    # Load model
    model = VFIformerSmall(cfgs)
    # normal
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/CKM/meta2/pretrain_scale/scale_2_4/haze/81_pretrain_net_40.pth',map_location='cpu')
    # ll
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/CKM/meta2/pretrain_scale/scale_8_16/ll/81_pretrain_net_30.pth',map_location='cpu')
    # haze
    # model_med = torch.load('/media/mygo/partition2/zzx/shizeru/CKM/meta2/pretrain_scale/scale_8_16/haze/81_pretrain_net_50.pth',map_location='cpu')
    # rain
    model_med = torch.load('/media/mygo/partition2/zzx/shizeru/CKM/meta2/pretrain_scale/scale_8_16/rain/81_pretrain_net_30.pth',map_location='cpu')
    
    model.load_state_dict(model_med['net'])
    model.to(device)
    model.eval()

    # Load validation dataset
    val_dataset = datasets.fetch_dataloader(cfgs)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Evaluate PME
    evaluate_SNet(model, val_loader, cfgs, device)