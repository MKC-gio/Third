import os
import sys
import time
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import copy
from functools import partial, reduce
import numpy as np
import itertools
import math
from encoder import RHWF_Encoder
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.warplayer import warp_f,warp
from models.transformer_layers import TFModel
from models.utils import coords_grid
from ATT.attention_layer import Correlation, FocusFormer_Attention
import torchgeometry as tgm
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )


def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.conv1 = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor= 1. / self.scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.conv1(x)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor= self.scale, mode="bilinear", align_corners=False)
        return flow

    
class IFNet(nn.Module):
    def __init__(self, args=None):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, scale=4, c=240)
        self.block1 = IFBlock(10, scale=2, c=150)
        self.block2 = IFBlock(10, scale=1, c=90)

    def forward(self, x):
        flow0 = self.block0(x)
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp_f(x[:, :3], F1_large[:, :2])
        warped_img1 = warp_f(x[:, 3:], F1_large[:, 2:4])
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = x[:,:3]
        warped_img1 = warp_f(x[:, 3:], F2_large[:, 2:4])
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
        F3 = (flow0 + flow1 + flow2)

        return F3, [F1, F2, F3]



    
class FlowRefineNet_Multis_Simple(nn.Module):
    def __init__(self, c=24):
        super(FlowRefineNet_Multis_Simple, self).__init__()

        self.conv1 = Conv2(3, c, 1)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2*c , 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8

        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        # out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:])

        return flow, c0, out1

    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp_f(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs


class Initialize_Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, b):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//b, W//b).cuda(0)
        coords1 = coords_grid(N, H//b, W//b).cuda(0)
        return coords0, coords1

class Conv1(nn.Module):
    def __init__(self, input_dim = 145):
        super(Conv1, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 1, padding=0, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x

class Conv3(nn.Module):
    def __init__(self, input_dim = 130):
        super(Conv3, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x

# class CNN_128(nn.Module):
#     def __init__(self):
#         super(CNN_128, self).__init__()

#         self.layer1 =nn.Sequential(nn.MaxPool2d(kernel_size = 2, stride=2),
#                                    nn.MaxPool2d(kernel_size = 2, stride=2),
#                                    nn.MaxPool2d(kernel_size = 2, stride=2),
#                                    nn.MaxPool2d(kernel_size = 2, stride=2),
#                                    nn.MaxPool2d(kernel_size = 2, stride=2),
#                                    nn.MaxPool2d(kernel_size = 2, stride=2))

#     def forward(self, x):
#         x = self.layer1(x)
#         return x
    
# class CNN_1128(nn.Module):
#     def __init__(self, input_dim=256):
#         super(CNN_1128, self).__init__()

#         outputdim = input_dim
#         self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
#                                     nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
#         input_dim = outputdim
#         outputdim = input_dim
#         self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
#                                     nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
#         input_dim = input_dim
#         outputdim = input_dim
#         self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
#                                     nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
#         input_dim = input_dim
#         outputdim = input_dim
#         self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
#                                     nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
#         input_dim = input_dim
#         outputdim = input_dim
#         self.layer5 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
#                                     nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
#         input_dim = input_dim
#         outputdim = input_dim
#         self.layer6 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
#                                     nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
#         outputdim_final = outputdim
#         # global motion
#         self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
#                                      nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.layer10(x)
#         return x


class CNN_32(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_32, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)
        return x
    
class CNN_16(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN_16, self).__init__()
        
        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(128, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim_final = outputdim
        # global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                    nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer10(x)
        return x


class GMA_update(nn.Module):
    def __init__(self, args, sz):
        super().__init__()
        self.args = args
        if sz==16:
            self.cnn = CNN_16(80)
        if sz==32:
            self.cnn = CNN_32(64)
        # if sz == 128:
        #     self.cnn = CNN_128()
        # if sz == 1128:
        #     self.cnn = CNN_1128(80)
            
    def forward(self, corr_flow):      
        delta_flow = self.cnn(corr_flow)   
        return delta_flow
    


    
class Get_Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sz, four_point,a):
        four_point = four_point/ torch.Tensor([a]).cuda(0)

        four_point_org = torch.zeros((2, 2, 2)).cuda(0)

        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, sz[2]-1]) 
        four_point_org[:, 1, 1] = torch.Tensor([sz[3]-1, sz[2]-1]) #四个点变换坐标
        # self.encoder = RHWF_Encoder(output_dim=96, norm_fn='instance')
        
        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(sz[0], 1, 1, 1)

        four_point_new = four_point_org + four_point

        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, sz[3]-1, steps=sz[3]), torch.linspace(0, sz[2]-1, steps=sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, sz[3] * sz[2]))),
                           dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)

        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(sz[0], sz[3], sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(sz[0], sz[3], sz[2]).unsqueeze(1)), dim=1)
        return flow

class ResidualBlock_R(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock_R, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class RHWF_Encoder(nn.Module):
    def __init__(self, output_dim=256, norm_fn='instance', dropout=0.0):
        super(RHWF_Encoder, self).__init__()
        
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        
        # 1/2
        self.layer1 = self._make_layer(56,stride = 2)
        # 1/4
        self.layer2 = self._make_layer(80,stride = 2)
        # 1/8   
        self.layer3 = self._make_layer(80,stride = 2)
        # 1/16
        self.layer4 = self._make_layer(80,stride = 2)
        
        self.conv_1 = nn.Conv2d(56, output_dim, kernel_size=1)
        self.conv_2 = nn.Conv2d(80, output_dim, kernel_size=1)
        self.conv_3 = nn.Conv2d(80, output_dim, kernel_size=1)
        self.conv_4 = nn.Conv2d(80, output_dim, kernel_size=1)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride):
        layer1 = ResidualBlock_R(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock_R(dim, dim, self.norm_fn, stride=1)      
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # 1/2   
        x = self.layer1(x)
        x_64 = self.conv_1(x)

        # 1/4
        x = self.layer2(x)
        x_32 = self.conv_2(x)
        
        # 1/8
        x = self.layer3(x)
        x_16 = self.conv_3(x)

        # 1/16
        x = self.layer4(x)
        x_8 = self.conv_4(x)
        

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x_16, x_32


class VFIformerSmall(nn.Module):
    def __init__(self,cfgs):
        super(VFIformerSmall, self).__init__()
        self.cfgs = cfgs
        # self.phase = args.phase
        self.device = torch.device('cuda:'+ str(self.cfgs.gpuid))
        c = 24
        height = 128
        width = 128
        window_size = 4
        embed_dim = 96
        self.conv3 = Conv3(input_dim=130)
        # self.update_block_raw = GMA_update(self.cfgs, 1128)
        # self.update_block = GMA_update(self.cfgs,128)
        # self.initialize_flow = Initialize_Flow()
        self.encoder = RHWF_Encoder(output_dim=96, norm_fn='instance')
        if self.cfgs.lev0:
            self.initialize_flow_8 = Initialize_Flow()
            self.transformer_0 = FocusFormer_Attention(96, 1, 96, 96)
            # self.kernel_list_0 = [0, 9, 5, 3, 3, 3] #此处0表示GM全局
            # self.pad_list_0    = [0, 4, 2, 1, 1, 1]
            sz = 16
            self.kernel_0 = 17
            self.pad_0 = 8
            self.conv1_0 = Conv1(input_dim=145)
            self.update_block_8 = GMA_update(self.cfgs, sz)
            
        if self.cfgs.lev1:
            self.initialize_flow_4 = Initialize_Flow()
            # self.kernel_list_1 = [5, 5, 3, 3, 3, 3]
            # self.pad_list_1    = [2, 2, 1, 1, 1, 1]
            sz = 32
            self.kernel_1 = 9
            self.pad_1 = 4
            self.conv1_1 = Conv1(input_dim=81)
            self.update_block_4 = GMA_update(self.cfgs, sz)

        # self.flownet = IFNet()
        # self.refinenet = FlowRefineNet_Multis(c=c, n_iters=1)
        # self.refinenet_sim = FlowRefineNet_Multis_Simple(c=c)
        # self.fuse_block = nn.Sequential(nn.Conv2d(6, c, 3, 1, 1),
        #                                 #  nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                                  nn.Conv2d(c, c, 3, 1, 1),
        #                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.transformer = TFModel(img_size=(height, width), in_chans=3, out_chans=4, fuse_c=c,
                                        window_size=window_size, img_range=1.,
                                        depths=[[3, 3], [3, 3], [3, 3], [1, 1]],
                                        embed_dim=embed_dim, num_heads=[[2, 2], [2, 2], [2, 2], [2, 2]], mlp_ratio=2,
                                        resi_connection='1conv',
                                        use_crossattn=[[[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [False, False, False, False]]])
                                                        # [[False, False, False], [True, True, True]], \
                                                        # [[False, False, False], [False, False, False]]])


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def get_flow(self, img0, img1):
    #     imgs = torch.cat((img0, img1), 1)
    #     flow, flow_list = self.flownet(imgs)
    #     flow, c0, c1 = self.refinenet(img0, img1, flow)

    #     return flow

    def forward(self,img0,img1,test_mode = False, outloop = 2, pretrain = False):
        
        # 检查并调整输入维度
        if img0.shape[1] != 3:  # 如果通道数不在第二个维度
            img0 = img0.permute(0, 3, 1, 2)  # 从 [B, H, W, C] 转换为 [B, C, H, W]
        if img1.shape[1] != 3:
            img1 = img1.permute(0, 3, 1, 2)

        img0 = 2 * (img0 / 255.0) - 1.0
        img1 = 2 * (img1 / 255.0) - 1.0
        img0 = img0.contiguous() #对张量进行连续重排
        img1 = img1.contiguous()
        img1_org = img1
        
        # warped_img1 = warp(img1, flow_med)
        # warped_img0 = warp_f(img0, flow_pre[:, :2])
        # warped_img1 = warp_f(img1, flow_pre[:, 2:])
        
        # four_point_predictions = []
        # four_point_predictions.append(four_point_disp)
        fmap1_16, fmap1_32 = self.encoder(img0) # [B, 96, 32, 32]
        fmap2_16, _        = self.encoder(img1) # [B, 96, 64, 64]
        # return fmap1_32

        four_point_disp = torch.zeros((img0.shape[0], 2, 2, 2)).cuda(0)
        four_point_predictions = []
        
        
        if self.cfgs.lev0:
            coords0, coords1 = self.initialize_flow_8(img0, 8) #生成网格在光流估计初始化使用
            coords0 = coords0.detach()
            self.get_flow_8 = Get_Flow()
            
            for itr in range(self.cfgs.iters_lev0):
                fea16_0, fea16_1 = self.transformer(fmap1_16, fmap2_16, 0) #（B,96,16,16）
                
                # fea16_0 = fmap1_16
                # fea16_1 = fmap2_16
                sz = fea16_0.shape
                coords1 = coords1.detach()
                corr = F.relu(Correlation.apply(fea16_0.contiguous(), fea16_1.contiguous(), self.kernel_0, self.pad_0))
                b, h, w, _ = corr.shape
                corr_1 = F.avg_pool2d(corr.view(b, h*w, self.kernel_0, self.kernel_0), 2).view(b, h, w, 64).permute(0, 3, 1, 2)
                corr_2 = corr.view(b, h*w, self.kernel_0, self.kernel_0)
                corr_2 = corr_2[:,:,4:13,4:13].contiguous().view(b, h, w, 81).permute(0, 3, 1, 2)                                                  
                corr = torch.cat([corr_1, corr_2], dim=1)
                corr = self.conv1_0(corr) 
                flow = coords1 - coords0                    
                corr_flow = torch.cat((corr, flow), dim=1)
                corr_flow = self.conv3(corr_flow)              
                delta_four_point = self.update_block_8(corr_flow)
                four_point_disp =  four_point_disp + delta_four_point
                four_point_predictions.append(four_point_disp) 
                coords1 = self.get_flow_8(sz,four_point_disp, 8)

                
                flow_med = coords1 - coords0
                flow_med = F.upsample_bilinear(flow_med, None, [8, 8]) *4     
                flow_med = flow_med.detach() 
                warped_img1 = warp(img1_org, flow_med)
                if itr < (self.cfgs.iters_lev0-1):
                    fmap2_64_warp, fmap2_32_warp, fmap2_16_warp, fmap2_8_warp  = self.encoder(warped_img1)
                    fmap2_16 = fmap2_16_warp.float()
                    
        # if test_mode and outloop == 0 and not pretrain:
        #     return fea16_1, four_point_predictions, warped_img1
        if test_mode and outloop == 0 and not pretrain:
            return fea16_1, four_point_predictions, warped_img1

        if self.cfgs.lev1:
            flow_med = coords1 - coords0
            flow_med = F.upsample_bilinear(flow_med, None, [8, 8]) * 4            
            flow_med = flow_med.detach()
            warped_img1 = warp(img1_org,flow_med)
            _, fmap2_32_warp = self.encoder(warped_img1)
            fmap2_32 = fmap2_32_warp.float()
            self.get_flow_4 = Get_Flow()
            
            coords0, coords1 = self.initialize_flow_4(img0, 4)
            coords0 = coords0.detach()
            
            for itr in range(self.cfgs.iters_lev1):
                # x0 = self.fuse_block(torch.cat([img0,img0], dim=1))
                # x1 = self.fuse_block(torch.cat([img1,warped_img1], dim=1))
                fea32_0 , fea32_1 = self.transformer(fmap1_32, fmap2_32)    #（B,96,32,32）
                
                # fea32_0 = fmap1_32
                # fea32_1 = fmap2_32
                sz = fea32_0.shape
                if iter == 0:
                    coords1 = self.get_flow_4(sz,four_point_disp, 4)
                
                coords1 = coords1.detach()
                corr = F.relu(Correlation.apply(fea32_0.contiguous(), fea32_1.contiguous(), self.kernel_1, self.pad_1)).permute(0, 3, 1, 2)    
                corr = self.conv1_1(corr)   
                flow = coords1 - coords0
                corr_flow = torch.cat((corr, flow), dim=1)
                corr_flow = self.conv3(corr_flow)  
                
                delta_four_point = self.update_block_4(corr_flow)
                four_point_disp = four_point_disp + delta_four_point
                four_point_predictions.append(four_point_disp)
                coords1 = self.get_flow_4(sz,four_point_disp, 4)

               
                flow_med = coords1 - coords0
                flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 2
                flow_med = flow_med.detach()
                warped_img1 = warp(img1_org, flow_med)
                if itr < (self.cfgs.iters_lev1-1):
                    _, fmap2_32_warp = self.encoder(warped_img1)
                    fmap2_32 = fmap2_32_warp.float()

        if test_mode and pretrain:
            return flow_med,four_point_predictions
        # elif test_mode and outloop == 1 and not pretrain:
        #     return fea32_1,four_point_predictions, warped_img1
        elif test_mode and outloop == 1 and not pretrain:
            return fea32_1,four_point_predictions, warped_img1
        elif test_mode and not pretrain:
        # elif test_mode and not pretrain:
            return four_point_predictions
        
        return four_point_disp,fea32_1


class Conv_Unit(nn.Module):
    def __init__(self,inc,out,k,d=1,cat_input=True):
        super(Conv_Unit,self).__init__()
        self.out = out
        self.k = k
        self.cat_input= cat_input
        
        self.Conv = nn.Sequential(*[nn.Conv2d(inc,out,k,bias=None,stride=1,padding=1),
                                    nn.ReLU()])
    def forward(self,x):
        f = self.Conv(x)
        if self.cat_input is True:
            mix = torch.concat([x,f],dim=1)
            self.conv = nn.Sequential(*[nn.Conv2d(mix.shape[1],self.out,self.k,bias=None,stride=1,padding=1),
                                        nn.ReLU()]).cuda(0)
            f = self.conv(mix)
            return f

        return f

class MFG(nn.Module):
    def __init__(self,out_channel=256,kernel=3,padding=1,d=1):
        super(MFG,self).__init__()
        self.out_channel = out_channel
        self.in_channel6 = 3
        self.in_channel2 = 3
        self.kernel = kernel
        # self.Conv6 = nn.Sequential(Conv_Unit(self.in_channel6,out_channel,kernel),
        #                            Conv_Unit(self.in_channel6*2,out_channel,kernel),
        #                            Conv_Unit(self.in_channel6*3,out_channel,kernel),
        #                            Conv_Unit(self.in_channel6*4,out_channel,kernel),
        #                            Conv_Unit(self.in_channel6*5,out_channel,kernel),
        #                            Conv_Unit(self.in_channel6*6,out_channel,kernel))
        
        self.Conv4 = nn.Sequential(Conv_Unit(96,128,kernel,False),
                                   Conv_Unit(128,256,kernel,False),
                                   Conv_Unit(256,512,kernel,False),
                                   Conv_Unit(512,256,kernel,False))
        # self.Conv2 = nn.Sequential(Conv_Unit(self.in_channel2,512,kernel,False),
        #                            Conv_Unit(512,256,kernel,False))
        
    
    def forward(self,fej,fuj):
        self.in_channel2 = fej.shape[1]
        f1 = self.Conv4(fuj)
        self.Conv2 = nn.Sequential(Conv_Unit(self.in_channel2,512,self.kernel,False),
                                   Conv_Unit(512,256,self.kernel,False)).cuda(0)
        f2 = self.Conv2(fej)
        f_m= torch.concat([f1,f2],dim=1)
        self.in_channel6 = f_m.shape[1]
        self.Conv6 = nn.Sequential(Conv_Unit(self.in_channel6,self.in_channel6*2,self.kernel),
                                #    Conv_Unit(self.in_channel6,self.in_channel6*2,self.kernel),
                                   Conv_Unit(self.in_channel6*2,self.in_channel6*2,self.kernel),
                                   Conv_Unit(self.in_channel6*2,self.in_channel6*2,self.kernel),
                                   Conv_Unit(self.in_channel6*2,self.in_channel6,self.kernel),
                                   Conv_Unit(self.in_channel6,self.out_channel,self.kernel)).cuda(0)
        f = self.Conv6(f_m)
        self.Conv = Conv_Unit(self.in_channel6,self.out_channel,k=3,cat_input=False).cuda(0)
        f_m = self.Conv(f_m)
        f = f + f_m
        return f
    
class FT(nn.Module):
    def __init__(self,in_channel,out_channel=256,kernel=3,padding=1,d=1):
        super(FT,self).__init__()
        self.Conv3 = nn.Sequential(Conv_Unit(in_channel,128,kernel,False),
                                   Conv_Unit(128,out_channel,kernel,False),
                                   Conv_Unit(256,out_channel,kernel,False))
        
    def forward(self,x):
        f = self.Conv3(x)
        return f
    


class Meta_block(nn.Module):
    def __init__(self,args):
        super(Meta_block,self).__init__()
        self.FT = FT(96)
        self.MFG = MFG()
    
    def Lg(self,ft,fm):
        # diff = ft - fm
        # square_diff = diff * diff
        # Lg = torch.sqrt(torch.sum(square_diff,dim = 1))
        Lg = torch.sqrt(torch.sum((ft - fm)**2))
        return Lg
    
    def forward(self,fej,fuj):
        ft = self.FT(fuj)
        fm = self.MFG(fej,fuj)
        ans = self.Lg(ft,fm)
        return ans,ft



# if __name__ == "__main__":
    # try:
    #     from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
    # except ImportError:
    #     raise ImportError('Failed to import DCNv2 module.')

    # import argparse
    # parser = argparse.ArgumentParser(description='test')
    # parser.add_argument('--phase', default='train', type=str)
    # parser.add_argument('--device', default='cuda', type=str)
    # parser.add_argument('--crop_size', default=192, type=int)
    # args = parser.parse_args()

    # device = 'cuda'

    # net = Swin_Fuse_CrossScaleV2_MaskV5_Normal_WoRefine_ConvBaseline(args).to(device)
    # print('----- generator parameters: %f -----' % (sum(param.numel() for param in net.parameters()) / (10**6)))
    
    # # w = 192
    # img0 = torch.randn((2, 3, w, w)).to(device)
    # img1 = torch.randn((2, 3, w, w)).to(device)
    # out = net(img0, img1)
    # print(out[0].size())


