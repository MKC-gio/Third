import os
import random
import shutil

# 定义文件夹路径
def sep():
    folder_path = '/media/mygo/partition4_hard/zzx/shizeru/rain_voc_label_20000/train/img_mask'  # 替换为你的文件夹路径

    # 列出文件夹中的文件
    file_list = sorted(os.listdir(folder_path))


    # 确定分割比例（这里示范的是一半一半）

    # 创建两个目标文件夹，用于存放数据的两份
    destination_folder_1 = '/media/mygo/partition4_hard/zzx/shizeru/rain_voc_meta_label/query/train/img_mask'  # 需要替换为你的第一个目标文件夹路径
    destination_folder_2 = '/media/mygo/partition4_hard/zzx/shizeru/rain_voc_meta_label/support/img_mask'  # 需要替换为你的第二个目标文件夹路径

    # 如果目标文件夹不存在，就创建它们


    # 将数据复制到不同的目标文件夹中
    for i, file_name in enumerate(file_list):
        source_file = os.path.join(folder_path, file_name)
        if i <=4000:
            destination = destination_folder_1
        else:
            destination = destination_folder_2
        shutil.copy(source_file, destination)

def choose():
   
 
    source_dir = '/media/mygo/partition2/zzx/shizeru/Meta-Homo/Data/Coordinate-v2'
    dest_dir = '/media/mygo/partition2/zzx/shizeru/Meta-Homo/Data/tmp'


    # for line in lines:
    #     line = line.strip()
    #     parts = line.split() 
    
    #     file1,file2 = parts[0],parts[1]

    #     if '0000085' in file1 or '0000091' in file1 or '0000092' in file1 or '00000100' in file1:
    #         file1 = file1.replace('LM', '')
    #         file2 = file2.replace('LM', '')
    #         file1 = '/media/mygo/partition4_hard/zzx/shizeru/CAHomo_raw/Test' + '/' + file1
    #         file2 = '/media/mygo/partition4_hard/zzx/shizeru/CAHomo_raw/Test' + '/' + file2
    #         shutil.copy(file1, lm1)
    #         shutil.copy(file2, lm2)
    #     else:
    #         file1 = file1.replace('LM', '')
    #         file2 = file2.replace('LM', '')
    #         file1 = '/media/mygo/partition4_hard/zzx/shizeru/CAHomo_raw/Test' + '/' + file1
    #         file2 = '/media/mygo/partition4_hard/zzx/shizeru/CAHomo_raw/Test' + '/' + file2
    #         shutil.copy(file1, out1)
    #         shutil.copy(file2, out2)
    
    # for root, dirs, files in os.walk(shift):
    #     for fi in files:
    #         file_path = os.path.join(root, fi)
    #         if '0000085' in fi or '0000091' in fi or '0000092' in fi or '00000100' in fi:
    #             shutil.copy(file_path, os.path.join(lms, fi))
    #         else:
    #             shutil.copy(file_path, os.path.join(outs, fi))
    # for filename in os.listdir(source_dir):
    #     # 检查文件名是否包含字母 "A"
    #     if '0000038' in filename:
    #     # if '0000085' in filename or '0000091' in filename or '0000092' in filename or '00000100' in filename:
    #         source_path = os.path.join(source_dir, filename)
    #         dest_path = os.path.join(dest_dir, filename)
            
    #         # 移动文件到目标文件夹
    #         shutil.copy(source_path, dest_path)
    


    folder_path = '/media/mygo/partition2/zzx/shizeru/Meta-Homo/Data/evaluation_files'  # 指定文件夹路径

    # 获取文件夹中所有文件的列表
    file_list = os.listdir(folder_path)

    # 遍历文件列表，删除文件名包含特定字符串的文件
    for file_name in file_list:
        if '0000085' in file_name or '0000092' in file_name or '00000100' in file_name:  # 根据具体需要修改匹配条件
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
    

    print("Deletion complete.")


if __name__ == '__main__':
    choose()