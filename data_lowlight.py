import cv2
import numpy as np
import os
import random as rd
def add_noise(image, noise_type='gaussian', mean=0, std_dev=25):
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'poisson':
        noisy_image = np.random.poisson(image / 255.0 * std_dev) / std_dev * 255
    else:
        raise ValueError("Unsupported noise type")
    return noisy_image

def simulate_low_light(image, exposure_reduction=0.5):
    low_light_image = (image * exposure_reduction).astype(np.uint8)
    return low_light_image

def add_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def adjust_contrast(image, contrast_factor=0.5):
    mean = np.mean(image)
    adjusted_image = ((image - mean) * contrast_factor + mean).astype(np.uint8)
    return adjusted_image

def adjust_brightness(image, brightness_factor=0.5):
    adjusted_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    return adjusted_image

# seqs=['V02', 'V03', 'V04']
# for seq in seqs:

if __name__ == '__main__':
    root= '/media/mygo/partition4_hard/zzx/shizeru/voc_test/input2'
    save_root='/media/mygo/partition4_hard/zzx/shizeru/voc_test_ll1/input2'
    os.makedirs(save_root,exist_ok=True)
    # 读取自然光图像
    # folders=sorted(os.listdir(root))
    # for folder in folders:
        # os.makedirs(os.path.join(save_root, folder),exist_ok=True)
    names = sorted(os.listdir(os.path.join(root)))
    for name in names:
        print(name)
        natural_light_image = cv2.imread(os.path.join(root,name))
        noisy_image = add_noise(natural_light_image, noise_type='gaussian', std_dev=rd.choice([25,26,27,28])) # 添加噪声
        low_light_image = simulate_low_light(noisy_image, exposure_reduction=rd.choice([0.8,0.9])) # 降低曝光度
        blurred_image = add_blur(low_light_image, kernel_size=(5, 5)) # 添加模糊
        contrast_adjusted_image = adjust_contrast(blurred_image, contrast_factor=rd.choice([0.5,0.6])) # 调整对比度
        brightness_adjusted_image = adjust_brightness(contrast_adjusted_image, brightness_factor=rd.choice([0.5,0.6])) # 调整亮度
        cv2.imwrite(os.path.join(save_root, name),brightness_adjusted_image)
            # cv2.imshow('Low Light Image', brightness_adjusted_image) # 显示合成的低光图像
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
