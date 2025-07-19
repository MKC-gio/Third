import numpy as np
import os
import cv2
import math
from numba import jit
import random
from glob import glob
import os.path as osp

# only use the image including the labeled instance objects for training
def load_annotations(annot_path):
    print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations


# print('*****************Add haze offline***************************')
def parse_annotation(len,path,dir):
    for num in range(0,len):
        print(num)
        # print(image_path)
        image_path = path[num]
        # print(image_path)
        img_name = image_path.split('/')[-1]
        # print(img_name)
        # print(img_name)
        # image_name = img_name.split('.')[0]
        # print(image_name)
        # image_name_index = img_name.split('.')[1]
        # print(image_name_index)
        # print(ssd)

    #'/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image %s" % image_path)
        for i in range(12):
            @jit()
            def AddHaz_loop(img_f, center, size, beta, A):
                (row, col, chs) = img_f.shape

                for j in range(row):
                    for l in range(col):
                        d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                        td = math.exp(-beta * d)
                        img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
                return img_f

            img_f = image/255
            (row, col, chs) = image.shape
            A = 0.5  
            # beta = 0.08  
            beta = 0.01 * i + 0.05
            size = math.sqrt(max(row, col)) 
            center = (row // 2, col // 2)  
            foggy_image = AddHaz_loop(img_f, center, size, beta, A)
            img_f = np.clip(foggy_image*255, 0, 255)
            img_f = img_f.astype(np.uint8)
        
        save_path = '/media/mygo/partition4_hard/zzx/shizeru/Homo_haze_try/' + dir + '/' 
        # print(save_path)
        os.makedirs(save_path, exist_ok=True)
        name = os.path.join(save_path, img_name)
        cv2.imwrite(name, img_f)

if __name__ == '__main__':
    an = '/media/mygo/partition4_hard/zzx/shizeru/HomoGr/input2'
    
    # subdirs = os.listdir(an)[1:]
    
    # for sub in os.listdir(an):
    #     if '0000085' in sub or '0000091' in sub or '0000092' in sub or '00000100' in sub or '0000038' in sub or '00000141' in sub or '0000044' in sub:
    #         continue
    dir = an.split('/')[-1]
    an = sorted(glob(osp.join(an, '*.jpg')))
        # subdir_path = os.path.join(an, sub)
        
        # path = sorted(glob(osp.join(subdir_path, '*.jpg')))
    l = len(an)
    parse_annotation(l,an,dir)

    
    
    # ans = sorted(glob(osp.join(tmp, '*.jpg')))
    # an2 = sorted(glob(osp.join(an2, '*.jpg')))
    # # h1 = sorted(glob(osp.join(h1, '*.jpg')))
    # # h2 = sorted(glob(osp.join(h2, '*.jpg')))
    # sh = sorted(glob(osp.join(shi, '*.npy')))
    # ll = len(an)
    # ll1 = len(an2)
    # l = len(tmp)
    # print(ll,ll1)
    # print(ssd)
    # # hl = len(h1)
    # # hl1 = len(h2)
    # ls = len(sh)
    # # print(ls)
    # # print(ll,l)
    # print(ls)
    # print(ssd)
    
    # # parse_annotation(ll,an,1)
    # parse_annotation(l,an2,2)

