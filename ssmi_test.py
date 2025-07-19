import cv2
import torch
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from math import log10
import numpy as np
import statistics
import torch.nn.functional as F
import math
import logging  # 导入logging库

# 设置日志配置
logging.basicConfig(filename='/media/mygo/partition2/zzx/shizeru/CKM/meta2/calculation_results.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()
    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def stdev(self):
        return statistics.stdev(self.vals)

def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)  
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0
    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True, channel_axis=2)
    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def ncc(img1, img2):
    return np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))

def mse(img1, img2):
    return np.sum((img1 - img2) ** 2) / len(img1)

def rmse(img1, img2):
    return np.sqrt(np.sum((img1 - img2) ** 2) / len(img1))

def gray(img1):
    return cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 记录多个路径的 PSNR、SSIM、NCC 值
def process_base_root(base_root, cond):
    val_psnr_full = AverageMeter()
    val_ssim_full = AverageMeter()
    val_ncc_full = AverageMeter()
    val_psnr = AverageMeter()
    val_ssim = AverageMeter()
    val_ncc = AverageMeter()

    names = sorted(os.listdir(os.path.join(base_root, 'input1')))
    i = 0
    for name in names:
        print(i)
        i += 1
        img1 = cv2.imread(os.path.join(base_root, 'input1', name))
        mask = cv2.imread(os.path.join(base_root, 'mask', name))/255
        img1 = img1 * mask
        img2 = cv2.imread(os.path.join(base_root, 'warp', name))
        temp_ncc = ncc(img1, img2)
        img1 = torch.from_numpy(img1).permute(2,0,1)[None]/255.
        img2 = torch.from_numpy(img2).permute(2,0,1)[None]/255.
        temp_psnr, temp_ssim, N = compute_psnr_ssim(img1, img2)
        if math.isnan(temp_ncc):
            temp_ncc = 0
        if temp_psnr < 100:
            val_psnr.update(temp_psnr, N)
            val_ssim.update(temp_ssim, N)
            val_ncc.update(temp_ncc, N)
            val_psnr_full.update(temp_psnr, N)
            val_ssim_full.update(temp_ssim, N)
            val_ncc_full.update(temp_ncc, N)

    # 输出到日志文件
    log_message = '{}: psnr:{}  ssmi:{}  ncc:{}'.format(
        cond, str(val_psnr.avg), str(val_ssim.avg), str(val_ncc.avg)
    )
    logging.info(log_message)

    # 打印输出结果
    print(log_message)


# 需要处理的路径列表
base_roots = [
                    #    #normal#    #
    # our
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/our/normal',
    # notran
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/notran/normal',
    # without semantical feature
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/normal/iters5',

    # iters3
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/normal/iters3',
    # iters4
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/normal/iters4',
    # iters5
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/normal/iters5',
    # iters6
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/normal/iters6',
    # iters7
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/normal/iters7',
    
                        #    #rain#    #
    # our
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/our/rain',
    # notran
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/notran/rain',
    # without semantical feature
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/notran/rain',

    # iters3
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/rain/iters3',
    # iters4
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/rain/iters4',
    # iters5
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/rain/iters5',
    # iters6
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/rain/iters6',
    # iters7
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/rain/iters7',
    
                        #    #ll#    #
    # our
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/our/ll',
    # notran
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/notran/ll',
    # without semantical feature
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/notran/ll',

    # iters3
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/ll/iters3',
    # iters4
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/ll/iters4',
    # iters5
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/ll/iters5',
    # iters6
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/ll/iters6',
    # iters7
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/ll/iters7',
    
    
                        #    #haze#    #
    # our
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/our/haze',
    # notran
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/notran/haze',
    # without semantical feature
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/notran/haze',

    # iters3
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/haze/iters3',
    # iters4
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/haze/iters4',
    # iters5
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/haze/iters5',
    # iters6
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/haze/iters6',
    # iters7
    '/media/mygo/partition4_hard/zzx/shizeru/PNSR_part2/CAHomo/haze/iters7'
        
]

# 对多个路径进行处理
for base_root in base_roots:
    cond = 'NN' 
    process_base_root(base_root, cond)