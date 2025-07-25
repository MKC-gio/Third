import cv2
import numpy as np
import glob
import os
import argparse

def get_noise(img, value=10):
    '''
    #生成噪声图像
    value= 大小控制雨滴的多少
    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    '''
    将噪声加上运动模糊,模仿雨滴

    noise:输入噪声图,shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle: 倾斜的角度，逆时针为正
    w:      雨滴大小

    '''

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


def alpha_rain(rain, img, img_name, out_dir, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    # 显示下雨效果
    # expand dimensin
    # 将二维雨噪声扩张为三维单通道
    # 并与图像合成在一起形成带有alpha通道的4通道图像
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()  # 拷贝一个掩膜
    rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）

    # cv2.imwrite(os.path.join(out_dir, os.path.basename(img_name)), rain_result)
    cv2.imwrite(out_dir, rain_result)


def process(img_name, out_dir, noise, rain_len, rain_angle, rain_thickness, alpha):
    img = cv2.imread(img_name)
    noise = get_noise(img, value=noise)
    rain = rain_blur(noise, length=rain_len, angle=rain_angle, w=rain_thickness)
    alpha_rain(rain, img, img_name, out_dir, beta=alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAPNet_train_test")
    parser.add_argument("--input_dir", type = str, default='C:/Users/Lebron/Desktop/CUSTOM_IMAGE/CityScape150/*.png')
    parser.add_argument("--output_dir", type = str, default='D:/Code/AAAI_2022/results/CityScape150_rain')
    parser.add_argument("--noise", type = int, default=500)
    parser.add_argument("--rain_len", type = int, default=50)
    parser.add_argument("--rain_angle", type = int, default=-30) # negative means rain streaks leans left
    parser.add_argument("--rain_thickness", type = int, default=3)
    parser.add_argument("--alpha", type = float, default=0.7)

    config = parser.parse_args()
    # root='POT/V02'
    # seqs=['V03','V04']
    # for seq in seqs:
    root = '/media/mygo/partition4_hard/zzx/shizeru/HomoGr/input2'
    save_root='/media/mygo/partition4_hard/zzx/shizeru/Homo_rain/input2'
    os.makedirs(save_root,exist_ok=True)
        # 读取自然光图像
    folders = sorted(os.listdir(root))
    # for folder in folders:
    #     print(folder)
    #     os.makedirs(os.path.join(save_root, folder),exist_ok=True)
    names = sorted(os.listdir(os.path.join(root)))
    for name in names:
        print(name)
        process(img_name = os.path.join(root, name),
                    out_dir = os.path.join(save_root, name),
                    noise = config.noise,
                    rain_len = config.rain_len,
                    rain_angle = config.rain_angle,
                    rain_thickness = config.rain_thickness,
                    alpha = config.alpha
                    )
        # for file in glob.glob(config.input_dir):
        #     process(img_name = file,
        #             out_dir = config.output_dir,
        #             noise = config.noise,
        #             rain_len = config.rain_len,
        #             rain_angle = config.rain_angle,
        #             rain_thickness = config.rain_thickness,
        #             alpha = config.alpha
        #             )









