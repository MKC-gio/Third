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
from glob import glob
from torch.cuda.amp import GradScaler
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from ssd import build_ssd
from layers.modules import MultiBoxLoss
import torch.utils.data as data
import importlib
from models.warplayer import warp
from models.VFIformer_arch import *
from models.utils import *
from utils.augmentations import SSDAugmentation
from models.utils import copy_to_device
from omegaconf import DictConfig, OmegaConf
from evaluate import validate_process
from data import *
from dataloader import dataset_new
from datasets_haze_voc import fetch_dataloader
from tqdm import tqdm

# os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Train:
    def __init__(self,cfgs):
        self.cfgs = cfgs
        self.curr_epoch = 1
        self.steps = 1
        self.step_index = 0
        self.total = 0
        self.device = torch.device('cuda:'+ str(self.cfgs.gpuid))
        # self.device = torch.device('cpu')
        # logging.info('Logs will be saved to %s' % self.cfgs.log_dir)
        # self.summary_writer = SummaryWriter(self.cfgs.log_dir)
        print('Loading ssd pre dataset')
        
        # self.pre_dataset =fetch_dataloader(self.cfgs.trainset)
        # self.preloader = torch.utils.data.DataLoader(
        #     dataset=self.pre_dataset,
        #     batch_size=1,
        #     num_workers=1,
        #     shuffle = False
        # )
        print('Loading support set')
        self.support_dataset =fetch_dataloader(self.cfgs.supset)
        self.support_loader = torch.utils.data.DataLoader(
            dataset=self.support_dataset,
            batch_size=self.cfgs.supset.batch_size,
            num_workers=self.cfgs.supset.num_works,
            shuffle = True,
            drop_last=True,
            pin_memory = True,
        )
        print('Loading train set in query set from %s' % self.cfgs.queset.trainset.dataset_root)
        self.train_dataset = fetch_dataloader(self.cfgs.queset.trainset)
        self.train_loader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                num_workers=self.cfgs.queset.trainset.num_works,
                batch_size=self.cfgs.queset.trainset.batch_size,
                shuffle=True,
                pin_memory = True,
            )
        print('Loading test set in quert set from %s' % self.cfgs.queset.testset.dataset_root)
        self.test_dataset = fetch_dataloader(self.cfgs.queset.testset)
        self.test_loader = torch.utils.data.DataLoader(
                dataset=self.test_dataset,
                batch_size=self.cfgs.queset.testset.batch_size,
                num_workers=self.cfgs.queset.testset.num_works,
                shuffle=True,
                pin_memory = True,
            )
        print('Loading val set  %s' % self.cfgs.queset.testset.dataset_root)
        self.val_dataset = fetch_dataloader(self.cfgs.queset.testset)
        self.val_loader = torch.utils.data.DataLoader(
                dataset=self.val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfgs.queset.testset.num_works,
                pin_memory = True,
            )
        
    
        print('********creating model*********')
        print('********creating Meta_block********')
        self.meta_model = Meta_block(self.cfgs)
        self.meta_model.to(self.device)
        # meta = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/res/metablock_0round.pth',map_location='cpu')
        self.weights_init('meta')
        self.optimizer_Meta = fetch_optimizer(self.cfgs.model.meta, self.meta_model)
        # self.meta_model.load_state_dict(meta['net'])
        # self.optimizer_Meta.load_state_dict(meta['optimizer'])

        print('********SSD_Model********')
        if self.cfgs.trainset.name == 'voc':
                self.cfg = voc
        self.ssd_net = build_ssd('train', self.cfg['min_dim'], self.cfg['num_classes'])
        print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
        # vgg_weight = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain/vgg16_reducedfc.pth')
        # self.ssd_net.vgg.load_state_dict(vgg_weight)
        # self.ssd_net.extras.apply(self.weights_init_ssd)
        # self.ssd_net.loc.apply(self.weights_init_ssd)
        # self.ssd_net.conf.apply(self.weights_init_ssd)
        self.ssd_net.load_weights('/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_haze_l5_truth/pre_ssd.pth')
        self.ssd_net.to(self.device)
        self.optimizerSSD = optim.SGD(self.ssd_net.parameters(), lr=self.cfgs.model.ssd.lr, momentum=self.cfgs.model.ssd.momentum,
                        weight_decay=self.cfgs.model.ssd.wdecay)
        self.criterion = MultiBoxLoss(self.cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                            False, self.device)
    
        print('********Fusion_Model********')
        self.fusion_model = VFIformerSmall(self.cfgs.model.fusion).to(self.device)
        self.optimizer_F, self.scheduler_F= fetch_optimizer_fusion(self.cfgs.model.fusion, self.fusion_model)
        # self.optimizer_F = fetch_optimizer_fusion(self.cfgs.model.fusion, self.fusion_model)
        # if self.cfgs.model.fusion.resume:
        
        # save_model = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_ll_a3/81_pretrain_net_20.pth',map_location='cpu')
        save_model = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_l5_haze_30000/81_pretrain_net_30.pth',map_location='cpu')
        # save_model = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_rain_a3/81_pretrain_net_100.pth',map_location='cpu')
        
        self.fusion_model.load_state_dict(save_model['net'])
        self.optimizer_F.load_state_dict(save_model['optimizer'])
        # else:
        # self.weights_init('fusion')
        self.amp_scaler = torch.cuda.amp.GradScaler()
        # log_file = '/media/mygo/partition2/zzx/shizeru/Meta-Homo/train/output.log'
        # logging.basicConfig(filename=log_file, level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
        # self.logger = logging.getLogger('output')
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('/media/mygo/partition2/zzx/shizeru/CKM/meta2/train_haze1/output.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def xavier(self,param):
        init.xavier_uniform(param)
    def weights_init_ssd(self,m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()
    def ssd_pre_warp(self):
        path = self.cfgs.trainset.image1
        path = sorted(glob(os.path.join(path, '*.jpg')))
        self.fusion_model.eval()
        for i_batch, inputs in enumerate(self.preloader):
            print(i_batch)
            img0 = inputs['img0'].to(self.device)
            img1 = inputs['img1'].to(self.device)
            flow_med,_ = self.fusion_model.forward(img0,img1,test_mode = True,pretrain=True)           
            image_warp = warp(img1,flow_med)
            image_warp = image_warp.detach().cpu().numpy()
            image_warp = np.transpose(image_warp,(0,2,3,1))
            image_warp =image_warp.squeeze(0)
            img0 = img0.detach().cpu().numpy()
            img0 = np.transpose(img0,(0,2,3,1))
            img0 = img0.squeeze(0)
            alpha = 0.5 
            beta = 1 - alpha
            img = cv2.addWeighted(img0, alpha, image_warp, beta, 0.0)
            
            save_path = self.cfgs.ssd_warp1
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path,path[i_batch].split('/')[-1].split('\\')[-1])
            # print(save_path)
            cv2.imwrite(save_path, img)
        print("dataset prepare done!!!!!")
    def fusion_img(self,img0,img1):
        alpha = 0.5
        beta = 1 - alpha
        
        result_img = img0 * alpha + img1 * beta
        return result_img
        
    def ssddataloader(self):
        self.ssd_dataset = VOCDetection(root=VOC_ROOT,transform=SSDAugmentation(self.cfg['min_dim'],
                                                         MEANS))
        self.ssd_dataloader = torch.utils.data.DataLoader(
                                dataset = self.ssd_dataset,
                                batch_size =self.cfgs.trainset.batch_size,
                                num_workers=self.cfgs.trainset.num_works,
                                shuffle=True, 
                                collate_fn=detection_collate,
                                pin_memory=True)
 
    def save_networks(self, round, epoch, last = False):
        if not last:
            save_filename = 'MetaHomo_{}round_{}epoch.pth'.format(round, epoch)
            save_path = os.path.join(self.cfgs.save_dir, save_filename)
            checkpoint = {
                "net": self.fusion_model.state_dict(),
                "optimizer": self.optimizer_F.state_dict(),
                # "scheduler": self.scheduler_F.state_dict()
            }
            torch.save(checkpoint, save_path)
        else:
            save_filename = 'MetaHomo_{}round.pth'.format(round)
            save_path1 = os.path.join(self.cfgs.save_ans, save_filename)
            checkpoint1 = {
                "net": self.fusion_model.state_dict(),
                "optimizer": self.optimizer_F.state_dict(),
                # "scheduler": self.scheduler_F.state_dict()
            }
            torch.save(checkpoint1, save_path1)
            meta_name = 'metablock_{}round.pth'.format(round)
            save_path = save_path = os.path.join(self.cfgs.save_ans, meta_name)
            checkpoint = {
                "net":self.meta_model.state_dict(),
                "optimizer": self.optimizer_Meta.state_dict()
            }
            torch.save(checkpoint, save_path)
                
    def get_learning_rate(self, cfgs, step,step_per_epoch):
        if step < 2000:
            mul = step / 2000.
        else:
            mul = np.cos((step - 2000) / (cfgs.max_epochs * step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return cfgs.lr * mul
                
    def trainssd_one_batch(self,iter):
        loss = 0
        loc_loss = 0
        conf_loss = 0
        for _, (images,targets) in enumerate(self.ssd_dataloader):
            if self.steps in self.cfgs.model.ssd.lr_steps:
                self.step_index +=1
                self.adjust_learning_rate(self.optimizerSSD,self.cfgs.model.ssd.gamma,self.step_index)
                
            if self.cfgs.gpuid:
                images = Variable(images.to(self.device)).to(torch.float32)
                targets = [Variable(ann.to(self.device),volatile = True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann,volatile = True) for ann in targets]
            
            out = self.ssd_net(images)
            self.optimizerSSD.zero_grad()
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            self.optimizerSSD.step()
            # print(loss_l.item(),loss_c.item(),loss.item())
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            self.total += loss.item()
            if self.steps % self.cfgs.model.ssd.print_step == 0:
                print('{} epoch {} steps: loc_loss: {}, conf_loss:{}, total_loss:{}'.format(iter+1, self.steps, loc_loss/100, conf_loss / 100,self.total/100))
                self.logger.info("{} epoch {} steps ssd model loss is: {}".format(iter+1,self.steps,self.total/100))
                loc_loss, conf_loss, self.total = 0,0,0
            
            self.steps +=1
            # loc_loss += loss_l.data[0]
            # conf_loss += loss_c.data[0]
        return loss
    
    
    def clone_model(self):
        model_clone = VFIformerSmall(self.cfgs.model.fusion).to(self.device)
        model_clone.load_state_dict(self.fusion_model.state_dict())
        return model_clone
    
    def trainfusion_one_batch(self):
        self.logger.info('begin {} epoch fusion train'.format(self.curr_epoch))
        step_per_epoch = self.support_loader.__len__()
        self.total = 0
        self.flow_total = 0
        
        for i, inputs in enumerate(self.support_loader):
            self.fusion_model.train()
            inputs = self.prepare(inputs)
            inputs = copy_to_device(inputs, self.device)
            img0 = inputs['img0']
            img1 = inputs['img1']
            flow_gt = inputs['flow_gt']
            gd = inputs['gd']
            
            # 确保gd是3通道的RGB图像
            if gd.shape[1] != 3:
                gd = gd[:, :3, :, :]  # 只取前3个通道
            
            # 将gd转换为浮点类型并调整大小
            gd = gd.float()
            gd = F.interpolate(gd, size=(300, 300), mode='bilinear', align_corners=False)
            
            # 获取语义特征
            fej = self.ssd_net.detect_feature(gd, 0)
            
            # 将语义特征传入fusion model
            fuj, pre_flow, image_warp = self.fusion_model.forward(img0, img1, fej, test_mode=True)
            
            # 计算损失
            flow_loss = sequence_loss(pre_flow, flow_gt, self.cfgs.model.fusion.gamma, self.cfgs.model.fusion)
            flow_loss.backward()

            # 更新fusion model
            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), self.cfgs.model.clip)
            self.optimizer_F.step()
            self.scheduler_F.step()
            self.optimizer_F.zero_grad()
            
            self.flow_total += flow_loss.item()
            self.total += flow_loss.item()
            
            print("{} epochs {} steps flow_loss is {} ".format(self.curr_epoch, self.steps, flow_loss.item()))
            
            if self.steps % self.cfgs.model.fusion.print == 0:
                print("{} epochs {} steps flow_loss is {} ".format(self.curr_epoch, self.steps, self.flow_total / 100))
                self.logger.info("{} epochs {} steps flow_loss is {} ".format(self.curr_epoch, self.steps, self.flow_total / 100))
                self.total = 0
                self.flow_total = 0
            
            self.steps += 1

    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if 'folder' not in key and 'pad_nums' not in key:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)

        return batch_samples

    def validate(self):
        self.fusion_model.eval()
        self.meta_model.eval()
        mace_list = []
        epoch_summary = None
     
        for i, inputs in enumerate(self.val_loader):
            print(i)
            inputs = self.prepare(inputs)
            inputs = copy_to_device(inputs, self.device)
            img0 = inputs['img0']
            img1 = inputs['img1']
            
            # 这部分注释可以保留，因为只是用于生成特征图
            gd = F.interpolate(img1, size=(300, 300), mode='bilinear', align_corners=False)
            fei = self.ssd_net.detect_feature(gd,1)
            
            # 需要取消这些注释，因为需要计算flow_gt和flow_4cor
            flow_gt = inputs['flow_gt']
            flow_4cor = torch.zeros((1, 2, 2, 2))
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
            
            # 需要取消这个注释，因为需要前向传播获取预测的flow
            with torch.cuda.amp.autocast(enabled=False):
                pre_flow, fuj = self.fusion_model.forward(img0,img1)
            
            # 这部分注释可以保留，因为只是用于生成特征图
            feature = fei.detach().cpu().numpy().squeeze(0)
            mean_feature_map= np.mean(feature, axis=0)
            feature_map = cv2.normalize(mean_feature_map, None, 0, 255, cv2.NORM_MINMAX)
            feature_map = np.uint8(feature_map)
            heatmap = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) 
            feature_map = cv2.resize(heatmap, (128, 128))
            cv2.imwrite('/media/mygo/partition4_hard/zzx/shizeru/ssd_fe/sfe/{}.jpg'.format(i+1), feature_map)
            
            # 需要取消这个注释，因为需要计算MACE值
            mace = torch.sum((pre_flow[0,:,:,:].cpu() - flow_4cor)**2, dim=0).sqrt()
            mace_list.append(mace.detach().view(-1).numpy())
            torch.cuda.empty_cache()
            
        self.fusion_model.train()
        mace = np.mean(np.concatenate(mace_list))
        print("Validation MACE: %f" % mace)
        return mace



    
    
    def weights_init(self,type,init_type = 'xavier', gain= 0.02):
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
        if type == 'fusion':
            self.fusion_model.apply(init_func)
        else:
            self.meta_model.apply(init_func)
    def adjust_learning_rate(self, optimizer, gamma, step):

        lr = self.cfgs.model.ssd.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        # print(self.optimizer_F.param_groups[0]['lr'])
        # print(ssd)
        # for i in range(0,self.cfgs.total_round):
        self.fusion_model.eval()
        self.meta_model.eval()
            # if i == 1:
            #     save_model = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_ll_l5_truth/MetaHomo_0round_40epoch.pth')
            #     self.fusion_model.load_state_dict(save_model['net'])
            #     self.optimizer_F.load_state_dict(save_model['optimizer'])
            #     meta_model = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/res/metablock_0round.pth')
            #     self.meta_model.load_state_dict(meta_model['net'])
            #     self.optimizer_Meta.load_state_dict(meta_model['optimizer'])
                # self.ssd_pre_warp()
        self.ssddataloader()
        # #     # if i == 1:
        # self.ssd_net.train()
        # for iteration in range(self.cfgs.model.ssd.epochs):
        #     loss = self.trainssd_one_batch(iteration)
        #     logging.info("{} iters ssd model loss is: {}".format(repr(iteration),loss))
        #                                             # print('iter:' + repr(iteration) + '|| Loss:%.4f||' %(loss),end = ' ')
        # torch.save(self.ssd_net.state_dict(),'/media/mygo/partition2/zzx/shizeru/Meta-Homo/train_haze_l5_truth/pre_ssd.pth')
        self.ssd_net.eval()
        
        self.steps = 1
        self.logger.info("begin fuison train")
        self.curr_epoch = 1
            # mace = self.validate()
            # self.logger.info('initial mace is {}:'.format(mace))
        mace = self.validate()
        self.logger.info('begin mace is {}:'.format(mace))
        while self.curr_epoch <= self.cfgs.model.fusion.max_epochs:
            self.trainfusion_one_batch()
                # logging.info('the {} epoch loss is {}'.format(self.curr_epoch, loss))
                # print("{} epochs flow_loss is {}".format(self.curr_epoch, loss))
            mace = self.validate()
            self.logger.info('{} epoch mace is {}:'.format(self.curr_epoch,mace))
                # self.summary_writer.add_scalar("loss", loss.item(), self.curr_epoch)
            if self.curr_epoch % self.cfgs.save_freq == 0:
                self.save_networks(0, self.curr_epoch)
            if self.curr_epoch % self.cfgs.model.fusion.max_epochs == 0:
                self.save_networks(0, self.curr_epoch, last=True)
            self.curr_epoch +=1



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--config', type=str, default='/media/mygo/partition2/zzx/shizeru/Meta-Homo/config/train_ll.yaml',
                        help='Path to the configuration (YAML format)')
    
    args = parser.parse_args()
    set_random_seed(0)
    
    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
        # cfgs.ckpt.path = args.weights
        # cfgs.model.fusion.resume = args.resume
        # if args.port != 0:
        #     cfgs.port = args.port

   
    # if os.path.exists(cfgs.log.dir) and not cfgs.model.fusion.resume:
    #     if input('Run "%s" already exists, overwrite it? [Y/n] ' % cfgs.log.run_name) == 'n':
    #         exit(0)
    #     shutil.rmtree(cfgs.log.dir, ignore_errors=True)
    # os.makedirs(cfgs.log.dir, exist_ok=True)
    # logging.basicConfig(filename='/media/mygo/partition2/zzx/shizeru/Meta-Homo/tmp.log', level=logging.INFO)
    # model1 = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain/81_pretrain_net_20.pth')
    # logging.info(model1)
    # model2 = torch.load('/media/mygo/partition2/zzx/shizeru/Meta-Homo/pretrain_11/81_pretrain_net_100.pth')
    # logging.basicConfig(filename='/media/mygo/partition2/zzx/shizeru/Meta-Homo/tmp1.log', level=logging.INFO)
    # logging.info(model2)
    # print(ssd)
    trainer = Train(cfgs)
    trainer.train()
