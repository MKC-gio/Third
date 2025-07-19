from tempfile import TemporaryFile
from PIL import Image
import numpy as np
import glob
import os
import random as rd
import cv2
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import torch



class_names = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class_dict = {class_name: i for i, class_name in enumerate(class_names)}


#Calculate cross product
def cross(a, b, c):
    # a, b, c = torch.tensor(a, device=device), torch.tensor(b, device=device), torch.tensor(c, device=device)
    ans = (b[0] - a[0])*(c[1] - b[1]) - (b[1] - a[1])*(c[0] - b[0])
    return ans.item()

#Check whether the quadrilateral is convex.
#If it is convex, return 1
def checkShape(a,b,c,d):
    # a, b, c, d = torch.tensor(a, device=device), torch.tensor(b, device=device), torch.tensor(c, device=device), torch.tensor(d, device=device)
    x1 = cross(a,b,c)
    x2 = cross(b,c,d)
    x3 = cross(c,d,a)
    x4 = cross(d,a,b)

    if (x1<0 and x2<0 and x3<0 and x4<0) or (x1>0 and x2>0 and x3>0 and x4>0) :
        return 1
    else:
        print('not convex')
        return 0


#Judge whether the pixel is within the label, and set it to black if it is not
def inLabel(row, col, src_input, dst):
    # src_input, dst = torch.tensor(src_input, device=device), torch.tensor(dst, device=device)
    # row, col = torch.tensor(row, device=device), torch.tensor(col, device=device)
    
    if (row >= src_input[0][1]) and (row <= src_input[2][1]) and (col >= src_input[0][0]) and (col <= src_input[1][0]) :
        return 1
    else : 
        #Only handle the case of convex quadrilaterals. As for the concave quadrilateral, regenerated it until it is convex.
        a = (dst[1][0] - dst[0][0])*(row - dst[0][1]) - (dst[1][1] - dst[0][1])*(col - dst[0][0])
        b = (dst[3][0] - dst[1][0])*(row - dst[1][1]) - (dst[3][1] - dst[1][1])*(col - dst[1][0])
        c = (dst[2][0] - dst[3][0])*(row - dst[3][1]) - (dst[2][1] - dst[3][1])*(col - dst[3][0])
        d = (dst[0][0] - dst[2][0])*(row - dst[2][1]) - (dst[0][1] - dst[2][1])*(col - dst[2][0])
        if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0) :
            return 1
        else :
            return 0





# Load a random image from the dataset
def load_random_image(path_source, size):
    #The size of the randomly sampled image must be greater than width*height
    img_path = rd.choice(glob.glob(os.path.join(path_source, '*.jpg'))) 
    img = Image.open(img_path)

    while True:
        flag = 0
        # print(img.size)
        if img.size[0]>=size[0] and img.size[1] >= size[1]:
            break
        img_path = rd.choice(glob.glob(os.path.join(path_source, '*.jpg'))) 
        img = Image.open(img_path)

    #print('bingo')

    img_grey = img.resize(size)  

        
    img_data = np.asarray(img_grey)
    #imggg = Image.fromarray(img_data.astype('uint8')).convert('RGB')
    #imggg.show()
    return img_data,img_path

def shift(bbox,src_input1,lap):
    
    x1 = float(bbox.find('xmin').text) - 1
    y1 = float(bbox.find('ymin').text) - 1
    x2 = float(bbox.find('xmax').text) - 1
    y2 = float(bbox.find('ymax').text) - 1

    
    if x2 <= src_input1[0][0] or y2 <= src_input1[0][1] and x1 >= src_input1[3][0] and y1 >= src_input1[3][1]:
        return False
    
    else:
        if x1 <= src_input1[0][0]:
            bbox.find('xmin').text = '0'
        else:
            bbox.find('xmin').text = str(x1 - src_input1[0][0])
        if y1 <= src_input1[0][1]:
            bbox.find('ymin').text = '0'
        else:
            bbox.find('ymin').text = str(y1 - src_input1[0][1])
        if x2 >= src_input1[3][0]:
            bbox.find('xmax').text = '128'
        else:
            bbox.find('xmax').text = str(x2 - src_input1[0][0])
        if y2 >= src_input1[3][1]:
            bbox.find('ymax').text = '128'
        else:
            bbox.find('ymax').text = str(y2 - src_input1[0][1])
            
    x1 = float(bbox.find('xmin').text)
    y1 = float(bbox.find('ymin').text)
    x2 = float(bbox.find('xmax').text)
    y2 = float(bbox.find('ymax').text)
    
    lx = max(x1,lap[0])
    ly = max(y1,lap[2])
    rx = min(x2,lap[1])
    ry = min(y2,lap[3])
    if lx < rx and ly < ry:
        bbox.find('xmin').text = str(lx)
        bbox.find('ymin').text = str(ly)
        bbox.find('xmax').text = str(rx)
        bbox.find('ymax').text = str(ry)
        return True
    else:
        return False
    
    
    # if x1 >= lap[2] or y1 >= lap[3] or x2 <= lap[0] or y2 <= lap[1]:
    #     return False
    # else:
    #     if x1 <= lap[0]:
    #         bbox.find('xmin').text = str(lap[0])
    #     else:
    #         bbox.find('xmin').text = str(x1 - lap[0])
    #     if y1 <= lap[1]:
    #         bbox.find('ymin').text = str(lap[1])
    #     else:
    #         bbox.find('ymin').text = str(y1 - lap[1])
    #     if x2 >= lap[2]:
    #         bbox.find('xmax').text = str(lap[2])
    #     else:
    #         bbox.find('xmax').text = str(x2 - lap[0])
    #     if y2 >= lap[3]:
    #         bbox.find('ymax').text = str(lap[3])
    #     else:
    #         bbox.find('ymax').text = str(y2 - lap[1])  
    #     return True
             

def _get_annotation(image_id,path,src_input,save_path,lap,index):
        annotation_file = os.path.join(path, "Annotations", "%s.xml" % image_id)
        # print(annotation_file)
        tmp_file = annotation_file
        tree = ET.parse(tmp_file)
        size = tree.find('size')
        size.find('width').text = '128'
        size.find('height').text = '128'
        target = tree.getroot()
        # for obj in 
        #     bbox = obj.find('bndbox')
        #     pts = ['xmin', 'ymin', 'xmax', 'ymax']
        # objects = tree.findall("object")
        # root = tree.getroot()
        # xml_string = ET.tostring(root,encoding='unicode')
        # print(xml_string)   

        boxes = []
        labels = []
        is_difficult = []
        flag1 = True
        for obj in target.findall('object'):
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            flag = shift(bbox,src_input,lap)
            # print(image_id)
            if int(obj.find('difficult').text) == 1:
                flag = False
            if flag is False:
                target.remove(obj)


        if target.find('object') is None:
            return False
        save_path = save_path + '/' + 'Annotation' + '/' + index + '.xml'
        tree.write(save_path) 
        return True
        # print(save_path)      
        # tr = ET.parse(save_path)
        # root = tr.getroot()
        # xml_string = ET.tostring(root,encoding='unicode')
        # print(xml_string)   
        # print(ssd)
            # VOC dataset format follows Matlab, in which indexes start from 0
            # x1 = float(bbox.find('xmin').text) - 1
            # y1 = float(bbox.find('ymin').text) - 1
            # x2 = float(bbox.find('xmax').text) - 1
            # y2 = float(bbox.find('ymax').text) - 1
            
        #     boxes.append([x1, y1, x2, y2])
        #     labels.append(class_dict[class_name])
        #     is_difficult_str = obj.find('difficult').text
        #     is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        # return (np.array(boxes, dtype=np.float32),
        #         np.array(labels, dtype=np.int64),
        #         np.array(is_difficult, dtype=np.uint8))

def get_annotation(index, path, apath,src_input1,lap,save_path):
        image_id = apath.split('/')[-1].split('\\')[-1].split('.')[0]
        ans = _get_annotation(image_id,path,src_input1,save_path,lap,index)
        if ans is False:
            return False
        else:
            return True
            
            


def save_to_file(index, image1, image2, label, path_dest, gt_4pt_shift):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    input1_path = path_dest +'//input1' 
    input2_path = path_dest +'//input2'
    label_path = path_dest +'//img_mask'
    pt4_shift_path = path_dest + '//shift'
    
    if not os.path.exists(input1_path):
        os.makedirs(input1_path)
    if not os.path.exists(input2_path):
        os.makedirs(input2_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    if not os.path.exists(pt4_shift_path):
        os.makedirs(pt4_shift_path)

    input1_path = path_dest +'//input1//' + index + '.jpg'
    input2_path = path_dest +'//input2//'+ index + '.jpg'
    label_path = path_dest +'//img_mask//'+ index + '.jpg'
    pt4_shift_path = path_dest +'//shift//'+ index + '.npy'
    image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
    image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
    label = Image.fromarray(label.astype('uint8')).convert('RGB')
    image1.save(input1_path)
    image2.save(input2_path)
    label.save(label_path)
    np.save(pt4_shift_path, gt_4pt_shift)


    

# Function to generate dataset
def generate_dataset(path_source, path_dest, rho, height, width, data, box, overlap):
    
    path = '/media/mygo/partition4_hard/zzx/shizeru/haze_VOC/'
    count = 0
    while True:
        if count >= data:
            break
        
        #load row image
        img,img_path= load_random_image(path_source, [width, height])
        img = img.astype(np.uint16)
        
        #define parameters
        #src_input1 = np.empty([4, 2], dtype=np.uint8)
        src_input1 = np.zeros([4, 2])
        src_input2 = np.zeros([4, 2])
        dst = np.zeros([4, 2])

        #Upper left
        src_input1[0][0] = int(width/2 - box/2)
        src_input1[0][1] = int(height/2 - box/2)
        # Upper right
        src_input1[1][0] = src_input1[0][0] + box
        src_input1[1][1] = src_input1[0][1]
        # Lower left
        src_input1[2][0] = src_input1[0][0]
        src_input1[2][1] = src_input1[0][1] + box
        # Lower right
        src_input1[3][0] = src_input1[1][0]
        src_input1[3][1] = src_input1[2][1]
        #print(src_input1)
        # print(src_input1)
        #The translation of input2 relative to input1


        # while True:
        box_x_off = rd.randint(int(box * (overlap - 1)), int(box * (1 - overlap)))
        box_y_off = rd.randint(int(box * (overlap - 1)), int(box * (1 - overlap)))
            #Upper left
        src_input2[0][0] = src_input1[0][0] + box_x_off
        src_input2[0][1] = src_input1[0][1] + box_y_off
            #Upper right
        src_input2[1][0] = src_input1[1][0] + box_x_off
        src_input2[1][1] = src_input1[1][1] + box_y_off
            # Lower left
        src_input2[2][0] = src_input1[2][0] + box_x_off
        src_input2[2][1] = src_input1[2][1] + box_y_off
            #Lower right
        src_input2[3][0] = src_input1[3][0] + box_x_off
        src_input2[3][1] = src_input1[3][1] + box_y_off
        

        offset = np.empty(8, dtype=np.int8)
        
        # Generate offsets:
        #The position of each vertex after the coordinate perturbation
        while True:
            for j in range(8):
                offset[j] = rd.randint(-rho, rho)
            # Upper left
            dst[0][0] = src_input2[0][0] + offset[0]
            dst[0][1] = src_input2[0][1] + offset[1]
            # Upper righ
            dst[1][0] = src_input2[1][0] + offset[2]
            dst[1][1] = src_input2[1][1] + offset[3]
            # Lower left
            dst[2][0] = src_input2[2][0] + offset[4]
            dst[2][1] = src_input2[2][1] + offset[5]
            # Lower right
            dst[3][0] = src_input2[3][0] + offset[6]
            dst[3][1] = src_input2[3][1] + offset[7]
            #print(dst)
            if checkShape(dst[0],dst[1],dst[3],dst[2])==1 :
                break

        source = np.zeros([4, 2])
        target = np.zeros([4, 2])
        source[0][0] = 0
        source[0][1] = 0
        source[1][0] = source[0][0] + box
        source[1][1] = source[0][1]
        source[2][0] = source[0][0]
        source[2][1] = source[0][1] + box
        source[3][0] = source[1][0]
        source[3][1] = source[2][1]
        target[0][0] = dst[0][0] - src_input1[0][0]
        target[0][1] = dst[0][1] - src_input1[0][1]
        target[1][0] = dst[1][0] - src_input1[0][0]
        target[1][1] = dst[1][1] - src_input1[0][1]
        target[2][0] = dst[2][0] - src_input1[0][0]
        target[2][1] = dst[2][1] - src_input1[0][1]
        target[3][0] = dst[3][0] - src_input1[0][0]
        target[3][1] = dst[3][1] - src_input1[0][1]
        # Hab, status = cv2.findHomography(source, target)
        
        

        # Generate the shift
        gt_4pt_shift = np.zeros((8,1), dtype = np.float32)
        for i in range(4):
            gt_4pt_shift[2*i] = target[i][0] - source[i][0]
            gt_4pt_shift[2*i+1] = target[i][1] - source[i][1]
        gt = np.zeros((2,2,2))
        gt[0, 0,:] = gt_4pt_shift[:2, 0]
        gt[0, 1,:] = gt_4pt_shift[2:4, 0]
        gt[1, 0,:] = gt_4pt_shift[4:6, 0]
        gt[1, 1,:] = gt_4pt_shift[6:8, 0]
        
        h, status = cv2.findHomography(dst, src_input2)
        img_warped = np.asarray(cv2.warpPerspective(img, h, (width, height))).astype(np.uint8)

        # Generate the label
          
        # Generate input1
        x1 = int(src_input1[0][0])
        y1 = int(src_input1[0][1])
        image1 = img[y1:y1+box, x1:x1+box]
        
        # Generate input2
        x2 = int(src_input2[0][0])
        y2 = int(src_input2[0][1])
        image2 = img_warped[y2:y2+box, x2:x2+box,...]
        
        
        
        x_min = int(max(src_input1[0][0], src_input2[0][0]))
        x_max = int(min(src_input1[1][0], src_input2[1][0]))
        y_min = int(max(src_input1[0][1], src_input2[0][1]))
        y_max = int(min(src_input1[2][1], src_input2[2][1]))
        
        img_mask = np.zeros_like(image1)
        over = image1[y_min-y1:y_max-y1, x_min-x1:x_max-x1]
        img_mask[y_min-y1:y_max-y1, x_min-x1:x_max-x1] = over
        # img_mask[int(src_input2[3][1])-y1:int(src_input1[0][1])-y1,int(src_input1[3][0])-x1:int(src_input2[0][0])-x1] = image1[int(src_input2[3][1])-y1:int(src_input1[0][1])-y1,int(src_input1[3][0])-x1:int(src_input2[0][0])-x1]
        lap = np.array([x_min-x1,x_max-x1,y_min-y1,y_max-y1])
        # img_mask[int(src_input1[0][1])-y1:int(lap[3])-y1,int(src_input1[0][0])-x1:int(lap[0])-x1] = 0
        # img_mask[int(lap[3])-y1:int(src_input1[3][1])-y1,int(src_input1[0][0])-x1:int(src_input1[1][0])-x1] = 0
        
        
        # flag = get_annotation(str(count+1).zfill(6),path,img_path,src_input1,lap,path_dest)
        # if flag is False:
        #     continue
        
        # path = '/media/mygo/partition4_hard/zzx/shizeru/haze_VOC/'
        
        # flag = get_annotation(str(count+1).zfill(6),path,img_path,src_input2,path_dest)
            

        save_to_file(str(count+1).zfill(6), image1, image2, img_mask, path_dest,gt)
        
        print(count+1)
        count +=1
        # print(ssd)
        



# raw_image_path = '/media/mygo/partition4_hard/zzx/shizeru/haze_VOC/ImageSet'       
raw_image_path = '/media/mygo/partition4_hard/zzx/shizeru/VOCdevkit/VOC2012/JPEGImages'  
box_size = 128
height = 360
width = 480
overlap_rate = 0.5
rho = int(box_size/5.0)

# ### generate training dataset
# print("Training dataset...")
# dataset_size = len([name for name in os.listdir(raw_image_path) if os.path.isfile(os.path.join(raw_image_path, name))])
# dataset_size = 20000
# generate_image_path = '/media/mygo/partition4_hard/zzx/shizeru/haze_voc_label_20000/train'
# # generate_image_path = '/media/mygo/partition4_hard/zzx/shizeru/try/train'
# generate_dataset(raw_image_path, generate_image_path, rho, height, width, dataset_size, box_size, overlap_rate)
# # generate testing dataset
print("Testing dataset...")
dataset_size = 2000
generate_image_path = '/media/mygo/partition4_hard/zzx/shizeru/voc_test'
generate_dataset(raw_image_path, generate_image_path, rho, height, width, dataset_size, box_size, overlap_rate)


