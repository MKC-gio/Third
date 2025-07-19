import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchgeometry as tgm
import os
import math
import random
from glob import glob
import os.path as osp
import scipy
import cv2
from torchvision import datasets, transforms
from skimage import io
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

marginal = 25
patch_size = 64

composed_transform1 = transforms.Compose([transforms.ToPILImage(),transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25), transforms.ToTensor(),transforms.RandomErasing(p=0.25, scale=(0.01, 0.1), ratio=(0.3, 3.3), value='random')])

class homo_dataset(data.Dataset):
    def __init__(self):

        self.is_test = False
        self.pretrain = False
        self.init_seed = True
        self.image_list_img1 = []
        self.image_list_img2 = []
        self.image_list_gt_image = []
        self.raw_image = ''
        self.si_img = ''
        self.label_list = []
        self.image_npy = []
        self.dataset = ''
        self.colorjit = False
        self.samples = dict()

    def __getitem__(self, index):
        if self.dataset == 'voc':
            img0 = cv2.imread(self.image_list_img1[index])
            img1 = cv2.imread(self.image_list_img2[index])
            # dimg = cv2.imread(self.image_list_gt_image[index])
            gd_img = img1.copy()
            # imgd = Image.open(self.image_list_img2[index])
            flow_gt = np.load(self.image_npy[index])
            flow_gt = torch.from_numpy(flow_gt).float().permute(2, 0, 1)
        
            h,w,c = img0.shape
            if self.is_test is False:
            # Randomly shift brightness
                if self.pretrain is False:
                    gd_img = cv2.imread(self.label_list[index])
                random_brightness = np.random.rand(1) * 0.6 + 0.7
                img0_aug = img0 * random_brightness
                random_brightness = np.random.rand(1) * 0.6 + 0.7
                img1_aug = img1 * random_brightness
                
                # Randomly shift color
                random_colors = np.random.uniform(0.7, 1.3, 3)
                white = np.ones(img0.shape)
                color_image = np.stack([white * random_colors[i] for i in range(3)], axis=3)
                # img0_aug  *= color_image
                for i in range(3):
                    img0_aug  *= color_image[:,:,:,i]


                random_colors = np.random.uniform(0.7, 1.3, 3)
                color_image = np.stack([white * random_colors[i] for i in range(3)], axis=3)
                # img1_aug  *= color_image
                for i in range(3):
                    img1_aug  *= color_image[:,:,:,i]

                
                img0 = img0_aug
                img1 = img1_aug
            

        elif self.dataset == 'CAHomo':
            name = self.image_npy[index]
            name = name.split('/')[-1]
            parts = name.split('_')
            if len(parts) > 2:
                first_part = '_'.join(parts[:2])
                second_part = '_'.join(parts[2:])
            if second_part.endswith('.npy'):
                second_part = second_part[:-len('.npy')]
            subfile = first_part.split('_')[0]
            img1_path = os.path.join(self.si_img, subfile, first_part+'.jpg',)
            img2_path = os.path.join(self.si_img, subfile, second_part+'.jpg')
            # img1_path = os.path.join(self.si_img, subfile, first_part)
            # img2_path = os.path.join(self.si_img, subfile, second_part)
            
            # print(img1_path)
            img0 = cv2.imread(img1_path)
            img1 = cv2.imread(img2_path)
            flow_gt = np.load(self.image_npy[index],allow_pickle=True)
            
            # src_pts = np.array([pt[0] for pt in flow_gt['matche_pts']], dtype=np.float32)
            # dst_pts = np.array([pt[1] for pt in flow_gt['matche_pts']], dtype=np.float32)
            # src_pts = np.array(flow_gt[0], dtype=np.float32)
            # dst_pts = np.array(flow_gt[1], dtype=np.float32)
            
            # H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            
            scale_x = 128 / 720
            scale_y = 128 / 480
            # src_pts_resized = src_pts * [scale_x, scale_y]
            # dst_pts_resized = dst_pts * [scale_x, scale_y]
            
            # H_resized, status = cv2.findHomography(src_pts_resized, dst_pts_resized, cv2.RANSAC)
            
            # height, width = 720, 480  # 根据你的图像尺寸调整
            # corners = np.float32([
            #     [0, 0],
            #     [width, 0],
            #     [0, height],
            #     [width, height],
            # ])
            # # .reshape(-1, 1, 2)
            # # corners1_homogeneous = np.hstack([corners, np.ones((4, 1), dtype='float32')])

   
        
        elif self.dataset == 'HomoGr':
            img0 = cv2.imread(self.image_list_img1[index])
            img1 = cv2.imread(self.image_list_img2[index])
            mat_data = scipy.io.loadmat(self.image_npy[index])
            # print(mat_data)
            data = mat_data['validation'][0,0]
            # print(data[1])
            wid,hei= data[1][0]
            # print(wid,hei)
            H = data[2]
            height, width = 128, 128
            scale_x = width / wid
            scale_y = height / hei
            scale_matrix = np.array([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ])
            H_mat = np.linalg.inv(scale_matrix) @ H @ scale_matrix
            H_mat = np.linalg.inv(H_mat)
            
            
            corners = np.float32([
                [0, 0],
                [width, 0],
                [0, height],
                [width, height],
            ]).reshape(-1, 1, 2)
            # print(H)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            # print(transformed_corners)
            # print(ssd)
            transformed_corners = transformed_corners - corners
            transformed_corners = transformed_corners.reshape(-1,2).reshape

            flow_gt = transformed_corners*[scale_x, scale_y]
            # flow_gt = flow_gt.reshape(-1,2).reshape(2,2,2)
            # print(flow_gt)
            # print(ssd)
            flow_gt = torch.from_numpy(flow_gt).float().permute(2, 0, 1)
            # print(flow_gt)
            # print(ssd)
            


            

            
            # x, y = np.meshgrid(np.arange(width), np.arange(height))
            # points = np.vstack([x.ravel(), y.ravel()]).T.reshape(-1, 1, 2).astype(np.float32)
            # print(points.shape)
            # print(ssd)

            # transformed_points = cv2.perspectiveTransform(points, H)
            # flow = transformed_points - points
            # flow_x = flow[:, :, 0].reshape(height, width)
            # flow_y = flow[:, :, 1].reshape(height, width)
            # pf_patch = np.zeros((128, 128, 2))
            # pf_patch[:, :, 0] = flow_x
            # pf_patch[:, :, 1] = flow_y
            # print(flow_gt)
            # print(ssd)
            



                        
        # img1_aug  = tf.clip_by_value(img1_aug,  -1, 1)
        # img2_aug  = tf.clip_by_value(img2_aug, -1, 1)
        

    
        
        if img0.shape[0] != 128:
            img0 = cv2.resize(img0,(128,128))
        if img1.shape[1] != 128:
            img1 = cv2.resize(img1,(128,128))
            
        img0 = torch.from_numpy(img0.astype('float32')).float().permute(2, 0, 1)
        img1 = torch.from_numpy(img1.astype('float32')).float().permute(2, 0, 1)
        # if self.pretrain is False and self.is_test is False:
        # gd_img = torch.from_numpy(gd_img.astype('float32')).float().permute(2, 0, 1)

        if self.is_test is False:
            if self.pretrain is False:
                sample = {'img0': img0,
                        'img1': img1,
                        'flow_gt': flow_gt,
                        'gd':gd_img}
            else:
                sample = {'img0': img0,
                    'img1': gd_img,
                    'flow_gt': flow_gt}
        else:
            sample = {'img0': img0,
                    'img1': img1,
                    # 'flow_gt':flow_gt}
                    'flow_gt1': flow_gt[0]*[scale_x,scale_y],
                    'flow_gt2': flow_gt[1]*[scale_x,scale_y]}


        return sample
    
    def transform_point(sefl,point, homography_matrix):
    # 将点转换为齐次坐标
        point_homogeneous = np.array([point[0], point[1], 1]).reshape((3, 1))
        
        # 使用单应矩阵进行变换
        transformed_point_homogeneous = np.dot(homography_matrix, point_homogeneous)
        
        # 将齐次坐标转换回二维坐标
        transformed_point = transformed_point_homogeneous / transformed_point_homogeneous[2]
        
        return transformed_point[0:2].flatten()
        
class TestDataset(Dataset):
    def __init__(self, benchmark_path):
        # self.input_transform = input_transform

        self.samples = np.load(benchmark_path, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1 = self.samples[idx]["img1"]
        img2 = self.samples[idx]["img2"]

        # gyro_homo = self.samples[idx]["homo"]

        gt_flow = self.samples[idx]["gt_flow"]
        gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1)

        split = self.samples[idx]["split"]

        # gyro_filed = transforms_lib.homo_to_flow(np.expand_dims(gyro_homo, 0), H=600, W=800)[0]

        # gyro_filed = gyro_filed.squeeze()

        # if self.input_transform:
        #     imgs_it = [self.input_transform(i) for i in imgs]
        if img1.shape[0] != 128:
            img1 = cv2.resize(img1,(128,128))
        if img2.shape[1] != 128:
            img2 = cv2.resize(img2,(128,128))
        
        img1 = torch.from_numpy(img1.astype('float32')).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2.astype('float32')).float().permute(2, 0, 1)

        ret = {"img0": img1, "img1": img2}
        # ret = {"img{}".format(i + 1): v for i, v in enumerate(imgs_it)}

        # ret["gyro_field"] = gyro_filed
        ret["gt_flow"] = gt_flow
        ret["label"] = split
        return ret

class MYDATA(homo_dataset):
    def __init__(self, cfgs,split='train', dataset='voc'):
        super(MYDATA, self).__init__()
        if split == 'train':
            if dataset=='voc':
                root_img1 = cfgs.image1
                root_img2 = cfgs.image2
                gd_img = cfgs.label
                shift = cfgs.shift
                self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.jpg'))) # 只使用前200张图片       
                self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.jpg')))
                self.label_list = sorted(glob(osp.join(gd_img, '*.jpg')))
                self.image_npy = sorted(glob(osp.join(shift, '*.npy')))
                
                # 检查数据长度并打印信息
                print(f"数据集长度检查:")
                print(f"image_list_img1: {len(self.image_list_img1)}")
                print(f"image_list_img2: {len(self.image_list_img2)}")
                print(f"label_list: {len(self.label_list)}")
                print(f"image_npy: {len(self.image_npy)}")

        else:
            if dataset=='voc':
                root_img1 = cfgs.image1
                root_img2 = cfgs.image2
                gd_img = cfgs.label
                shift = cfgs.shift
                self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.jpg')))   # 只使用前200张图片
                self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.jpg')))
                self.image_npy = sorted(glob(osp.join(shift, '*.npy')))
                
                # 检查数据长度并打印信息
                print(f"测试集长度检查:")
                print(f"image_list_img1: {len(self.image_list_img1)}")
                print(f"image_list_img2: {len(self.image_list_img2)}")
                print(f"image_npy: {len(self.image_npy)}")
            
            elif dataset == 'CAHomo':
                img_root = cfgs.img
                shift = cfgs.shift
                self.si_img = img_root
                self.image_npy = sorted(glob(osp.join(shift, '*.npy')))  # 只使用前200张图片
            elif dataset == 'HomoGr':
                root_img1 = cfgs.image1
                root_img2 = cfgs.image2
                shift = cfgs.shift
                self.image_list_img1 = sorted(glob(osp.join(root_img1, '*g')))      # 只使用前200张图片
                self.image_list_img2 = sorted(glob(osp.join(root_img2, '*g')))
                self.image_npy = sorted(glob(osp.join(shift, '*.mat')))
                
                
        self.colorjit = False
        self.dataset = dataset
        if split == 'train':
            self.is_test = False
            if cfgs.pretrain is True:
                self.pretrain = True
        else:
            self.is_test = True
            
    def __len__(self):
        if self.dataset == 'voc' or self.dataset == 'HomoGr':
            return int(len(self.image_list_img1))
        elif self.dataset == 'CAHomo':
            return int(len(self.image_npy))


def fetch_dataloader(cfgs):
    if cfgs.phase == 'train':
        train_dataset = MYDATA(cfgs,split= cfgs.phase, dataset=cfgs.name)
    else: 
        if cfgs.name == 'GHOF':
            train_dataset = TestDataset(benchmark_path=cfgs.img)
        else:
            train_dataset = MYDATA(cfgs,split= cfgs.phase,dataset=cfgs.name)
    return train_dataset

