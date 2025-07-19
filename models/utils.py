import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy import interpolate
from skimage import io
import random
import sys
import matplotlib.pyplot as plt


def copy_to_device(inputs, device, non_blocking=True):
    if isinstance(inputs, list):
        inputs = [copy_to_device(item, device, non_blocking) for item in inputs]
    elif isinstance(inputs, dict):
        inputs = {k: copy_to_device(v, device, non_blocking) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError('Unknown type: %s' % str(type(inputs)))
    return inputs

def sequence_loss(four_preds, flow_gt, gamma, args):
    """ Loss function defined over sequence of flow predictions """
    
    # 打印输入张量的维度信息
    print("Debug info:")
    print(f"four_preds length: {len(four_preds)}")
    print(f"iters_lev0: {args.iters_lev0}, iters_lev1: {args.iters_lev1}")
    print(f"four_preds[0] shape: {four_preds[0].shape}")
    print(f"flow_gt shape before processing: {flow_gt.shape}")
    
    # 检查输入维度
    if flow_gt.dim() == 3:
        flow_gt = flow_gt.unsqueeze(0)  # 添加 batch 维度
        print(f"flow_gt shape after unsqueeze: {flow_gt.shape}")
    
    # 确保 flow_gt 有正确的维度
    if flow_gt.shape[0] == 0:
        print("Warning: flow_gt has zero batch size")
        return torch.tensor(0.0, requires_grad=True).to(four_preds[0].device)
    
    # 创建 flow_4cor 并确保维度匹配
    flow_4cor = torch.zeros((four_preds[0].shape[0], 2, 2, 2), requires_grad=True).to(four_preds[0].device)
    print(f"flow_4cor shape: {flow_4cor.shape}")
    
    # 如果 batch size 不匹配，进行复制
    if flow_gt.shape[0] != flow_4cor.shape[0]:
        print(f"Adjusting batch size from {flow_gt.shape[0]} to {flow_4cor.shape[0]}")
        indices = torch.arange(flow_4cor.shape[0], device=flow_gt.device) % flow_gt.shape[0]
        flow_gt = flow_gt[indices]
        print(f"flow_gt shape after indexing: {flow_gt.shape}")
    
    # 确保 flow_gt 有正确的空间维度
    if flow_gt.shape[2] < 2 or flow_gt.shape[3] < 2:
        print(f"Warning: flow_gt spatial dimensions too small: {flow_gt.shape}")
        return torch.tensor(0.0, requires_grad=True).to(four_preds[0].device)
    
    try:
        # 填充四个角点
        flow_4cor[:,:, 0, 0] = flow_gt[:,:, 0, 0]
        flow_4cor[:,:, 0, 1] = flow_gt[:,:, 0, -1]
        flow_4cor[:,:, 1, 0] = flow_gt[:,:, -1, 0]
        flow_4cor[:,:, 1, 1] = flow_gt[:,:, -1, -1]
    except Exception as e:
        print(f"Error during corner assignment: {str(e)}")
        print(f"flow_gt shape at error: {flow_gt.shape}")
        print(f"flow_4cor shape at error: {flow_4cor.shape}")
        return torch.tensor(0.0, requires_grad=True).to(four_preds[0].device)

    ce_loss = 0.0
    # 确保不超出 four_preds 的长度
    actual_iters_lev0 = min(args.iters_lev0, len(four_preds))
    actual_iters_lev1 = min(args.iters_lev1, len(four_preds) - actual_iters_lev0)
    
    for i in range(actual_iters_lev0):
        i_weight = gamma**(actual_iters_lev0 - i - 1)
        # 调整 four_preds[i] 的维度以匹配 flow_4cor
        pred_reshaped = four_preds[i].view(four_preds[i].shape[0], 2, -1)
        pred_corners = torch.stack([
            pred_reshaped[:,:, 0],  # 左上角
            pred_reshaped[:,:, -1],  # 右上角
            pred_reshaped[:,:, 128],  # 左下角
            pred_reshaped[:,:, -1]  # 右下角
        ], dim=2).view(four_preds[i].shape[0], 2, 2, 2)
        
        i4cor_loss = (pred_corners - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()
        
    for i in range(actual_iters_lev0, actual_iters_lev0 + actual_iters_lev1):
        i_weight = gamma ** (actual_iters_lev1 + actual_iters_lev0 - i - 1)
        # 调整 four_preds[i] 的维度以匹配 flow_4cor
        pred_reshaped = four_preds[i].view(four_preds[i].shape[0], 2, -1)
        pred_corners = torch.stack([
            pred_reshaped[:,:, 0],  # 左上角
            pred_reshaped[:,:, -1],  # 右上角
            pred_reshaped[:,:, 128],  # 左下角
            pred_reshaped[:,:, -1]  # 右下角
        ], dim=2).view(four_preds[i].shape[0], 2, 2, 2)
        
        i4cor_loss = (pred_corners - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()

    return ce_loss





def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, epochs=50,steps_per_epoch=1000,
    #                                         pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer

def fetch_optimizer_fusion(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, epochs=40, steps_per_epoch=3425,
                                            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, epochs=100, steps_per_epoch=1250,
    #                                         pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer,scheduler
    # return optimizer


# def fetch_optimizer(cfgs, model,lr):
#     """ Create the optimizer and learning rate scheduler """
#     optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=cfgs.wdecay, eps=cfgs.epsilon)

#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr,
#                                             pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
#     return optimizer, scheduler


class Logger_(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)

        # logging running loss to total loss
        self.train_mace_list.append(np.mean(self.running_loss_dict['mace']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def save_img(img, path):
    npimg = img.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    io.imsave(path, npimg)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def warp(x, flo):
#     """
#     warp an image/tensor (im2) back to im1, according to the optical flow
#     x: [B, C, H, W] (im2)
#     flo: [B, 2, H, W] flow
#     """
#     B, C, H, W = x.size()
#     # mesh grid
#     xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
#     yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
#     xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     grid = torch.cat((xx, yy), 1).float()

#     if x.is_cuda:
#         grid = grid.cuda()
#     vgrid = torch.autograd.Variable(grid) + flo

#     # scale grid to [-1,1]
#     vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
#     vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

#     vgrid = vgrid.permute(0, 2, 3, 1)
#     output = nn.functional.grid_sample(x, vgrid, align_corners=True)
#     mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
#     mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

#     mask[mask < 0.999] = 0
#     mask[mask > 0] = 1

#     return output * mask

