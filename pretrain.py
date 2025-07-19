from __future__ import print_function, division
import json
import sys
import argparse
import logging
import os
import yaml
import shutil
import cv2
import time
import torch
import torchvision
from torch.cuda.amp import GradScaler
import torch.nn.init as init
from torch.autograd import Variable
from models.VFIformer_arch import VFIformerSmall
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from models.losses import PerceptualLoss, AdversarialLoss, EPE, Ternary
from models.utils import *
from omegaconf import DictConfig, OmegaConf
from datasets_haze_voc import fetch_dataloader
from collections import OrderedDict
import math
# from models_old.rhwf import RHWF

from evaluate import validate_process

# os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class Train:
    def __init__(self,cfgs):
        self.cfgs = cfgs
        self.curr_epoch,self.curr_step = 1, 1
        self.device = torch.device("cuda:0")
        print('Logs will be saved to %s' % self.cfgs.log_dir)
        # self.summary_writer = SummaryWriter(self.cfgs.log_dir)
    
        print('Loading training set %s' % self.cfgs.trainset.name)
        self.train_dataset = fetch_dataloader(self.cfgs.trainset)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.trainset.batch_size,
            num_workers=self.cfgs.trainset.num_works,
            # shuffle = True,
            pin_memory=True,
            drop_last=True,
        )
    
        print('Loading val set  %s' % self.cfgs.testset.name)
        self.val_dataset = fetch_dataloader(self.cfgs.testset)
        self.val_loader = torch.utils.data.DataLoader(
                dataset=self.val_dataset,
                batch_size=self.cfgs.testset.batch_size,
                num_workers = 8,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
            )
        print('********Fusion_Model********')
        self.fusion_model = VFIformerSmall(self.cfgs).to(self.device)
        self.optimizer_F,self.scheduler_F = fetch_optimizer_fusion(self.cfgs, self.fusion_model)
        # self.optimizer_F = fetch_optimizer_fusion(self.cfgs, self.fusion_model)
        
        # save_model = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_rain_a3/81_pretrain_net_40.pth',map_location='cpu')
        # self.fusion_model.load_state_dict(save_model['net'])
        # self.optimizer_F.load_state_dict(save_model['optimizer'])
        
        # if self.cfgs.resume:
        # save_model = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain/81_pretrain_net_20.pth')
        # self.fusion_model.load_state_dict(save_model['net'])
            # self.optimizer_F.load_state_dict(save_model['optimizer'])
            # self.load_networks('net', self.cfgs.resume)  
        # else:
        
        self.weights_init()    
    
    
    def weights_init(self,init_type = 'xavier', gain= 0.02):
        def init_func(net):
                for name, m in net.named_modules():
                    classname = m.__class__.__name__
                    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                        if init_type == 'normal':
                            init.normal_(m.weight.data, 0.0, gain)
                        elif init_type == 'xavier':
                            init.xavier_normal_(m.weight.data, gain=gain)
                        elif init_type == 'kaiming':
                            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                        elif init_type == 'orthogonal':
                            init.orthogonal_(m.weight.data, gain=gain)
                        else:
                            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                        if hasattr(m, 'bias') and m.bias is not None:
                            init.constant_(m.bias.data, 0.0)
                    elif classname.find('BatchNorm2d') != -1:
                        init.normal_(m.weight.data, 1.0, gain)
                        init.constant_(m.bias.data, 0.0)
        
        self.fusion_model.apply(init_func)
   
    def load_networks(self, net_name, resume, strict=True):
        load_path = resume
        load_net = torch.load(load_path)
        self.fusion_model.load_state_dict(load_net)
    def save_networks(self, net_name, epoch):
        save_filename = '81_{}_{}.pth'.format(net_name, epoch)
        save_path = os.path.join(self.cfgs.save_dir, save_filename)
        checkpoint = {
            "net": self.fusion_model.state_dict(),
            "optimizer": self.optimizer_F.state_dict()
        }
        torch.save(checkpoint, save_path)
        
    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if 'folder' not in key and 'pad_nums' not in key:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)

        return batch_samples

                
    def get_learning_rate(self, cfgs, step,step_per_epoch):
        if step < 2000:
            mul = step / 2000.
        else:
            mul = np.cos((step - 2000) / (cfgs.max_epoch * step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return cfgs.lr * mul
    
    def train(self):
        # log_file = '/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_ll_a3/mace.log'
        log_file = '/media/mygo/partition2/zzx/shizeru/CKM/meta2/pretrain_scale/scale_2_4/rain1/mace.log'
        logging.basicConfig(filename=log_file, level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
        # self.validate()
        # print(ssd)
        # mace = self.validate()
        # logging.info('begin mace is {}:'.format(mace))
        while self.curr_epoch <= self.cfgs.max_epoch:
            print("******* now is {} epochs".format(self.curr_epoch))
            self.trainfusion_one_batch()
            # if self.curr_epoch % 5 == 0 and self.curr_epoch != 0:
            mace = self.validate()
            logging.info('{} epoch mace is {}:'.format(self.curr_epoch,mace))
            if self.curr_epoch % self.cfgs.save == 0:
                self.save_networks('pretrain_net', self.curr_epoch)
            
            self.curr_epoch +=1
    # def flow_loss(self, flow_pre, flow_gt):
    #     flow_loss = 0
    #     for level in range(len(flow_pre)):
    #         fscale = flow_pre[level].size(-1) / flow_gt.size(-1)
            
    #         flow_gt_resize = F.interpolate(flow_gt, scale_factor=fscale, mode="bilinear",
    #                                     align_corners=False) * fscale
    #         flow_loss = self.criterion_flow(flow_pre[level], flow_gt_resize, 1).mean()
    #     flow_loss = flow_loss * self.lambda_flow
        
    #     return flow_loss
                
        
    def trainfusion_one_batch(self):
        self.fusion_model.train()
        step_per_epoch = self.train_loader.__len__()
        # log_file = '/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain/output.log'
        # logging.basicConfig(filename=log_file, level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('begin to train {} epoch'.format(self.curr_epoch))
        total_loss = 0
        for i, inputs in enumerate(self.train_loader):
            inputs = copy_to_device(inputs, self.fusion_model.device)
            inputs = self.prepare(inputs)
            img0 = inputs['img0']
            img1 = inputs['img1']
            flow_gt = inputs['flow_gt']
            # with torch.cuda.amp.autocast(enabled=self.cfgs.amp):
            _,pre_flow=self.fusion_model.forward(img0,img1,test_mode=True,pretrain=True)
            flow_loss = sequence_loss(pre_flow,flow_gt,self.cfgs.gamma,self.cfgs)
            total_loss += flow_loss
            
            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), self.cfgs.clip)
            self.optimizer_F.zero_grad()
            flow_loss.backward()
            self.optimizer_F.step()
            self.scheduler_F.step()

            # lrate = self.get_learning_rate(self.cfgs, self.curr_step,step_per_epoch)
            for param_group in self.optimizer_F.param_groups:
                    param_group['lr']
                    
            self.curr_step += 1
            
            if self.curr_step % 100 == 0 and self.curr_step != 0:
                loss = total_loss / 100
                total_loss = 0
                # for param_group in self.optimizer_F.param_groups:
                #     print(param_group['lr'])
                print("{} epochs {} steps flow_loss is {}".format(self.curr_epoch,self.curr_step, loss))
                logging.info("{} epochs {} steps flow_loss is {}".format(self.curr_epoch,self.curr_step, loss))
       
        # return flow_loss
    
    def validate(self):
        self.fusion_model.eval()
        # print(f"Validation dataset length: {len(self.val_dataset)}") 
        mace_list = []
        for i, inputs in enumerate(self.val_loader):
            # print(f"Validating sample {i}/{len(self.val_loader)}") 
            print(i)
            inputs = self.prepare(inputs)
            inputs = copy_to_device(inputs, self.device)
            img0 = inputs['img0'].cuda(0)
            img1 = inputs['img1'].cuda(0)
            flow_gt = inputs['flow_gt'].cuda(0)
            # print(f"flow_gt.shape: {flow_gt.shape}")  
            # flow_gt = flow_gt.squeeze(0)
            flow_4cor = torch.zeros((1, 2, 2, 2), device=self.device)
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
            # with torch.cuda.amp.autocast(enabled=False):
            # print("pre_flow type:", type(pre_flow))  
            pre_flow,_=self.fusion_model.forward(img0,img1)
            # print("pre_flow shape:", pre_flow.shape)
            # print("flow_4cor shape:", flow_4cor.shape)
            
            # pre_flow = self.fusion_model.forward(img0,img1)
            mace = torch.sum((pre_flow[0,:,:,:].to(self.device) - flow_4cor) ** 2, dim=0).sqrt()
            mace_list.append(mace.view(-1).detach().cpu().numpy())
            torch.cuda.empty_cache()
            #if i == 2000:
            #    break
        mace = np.mean(np.concatenate(mace_list))
        print("Validation MACE: %f" % mace)
        return mace


    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='/media/mygo/partition2/zzx/shizeru/CKM/meta2/config/pretrain_rain.yaml',
                        help='Path to the configuration (YAML format)')
    parser.add_argument('--weights', required=False, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--resume', required=False, action='store_true',
                        help='Resume unfinished training')
    parser.add_argument('--port', required=False, type=int, default=0,
                        help='Resume unfinished training')
    args = parser.parse_args()

    # load config
    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
        # cfgs.ckpt.path = args.weights
        # cfgs.ckpt.resume = args.resume
        # if args.port != 0:
        #     cfgs.port = args.port

    # set num_workers of data loader
    set_random_seed(0)
    


    # create log dir
    if os.path.exists(cfgs.log_dir) and not cfgs.resume:
        # if input('Run "%s" already exists, overwrite it? [Y/n] ' % cfgs.log.run_name) == 'n':
        #     exit(0)
        shutil.rmtree(cfgs.log_dir, ignore_errors=True)
    os.makedirs(cfgs.log_dir, exist_ok=True)
    

    trainer = Train(cfgs)
    trainer.train()
