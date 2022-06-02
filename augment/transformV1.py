import numpy as np
import cv2
import random

def rand(a,b):
    return np.random.rand() * (b - a) + a

def get_random_size(image, conf):
    '''随机缩放图片大小'''
    if conf.random_scale and conf.random_scale_p < random.random():
        ih, iw, _ = image.shape
        w = conf.image_w
        h = conf.image_h
        jitter = conf.jitter
        new_ar = w/h * rand(1-jitter, 1+jitter) /rand(1-jitter, 1+jitter)
        scale = rand(conf.scale_min, conf.scale_max)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = cv2.resize(image, (nw,nh))
        if conf.random_scale_show:
            cv2.imshow('1', image)
            cv2.waitKey(0)
    return image

def HorizonFilp(image, conf):
    '''左右翻转'''
    if conf.random_HorFilp and conf.random_HorFilp_p < random.random():
        image = image[:, ::-1, :]
        if conf.random_HorFilp_show:
            cv2.imshow('1', image)
            cv2.waitKey(0)
    return image

def VerFlip(image, conf):
    '''水平翻转'''
    if conf.random_VerFlip and conf.random_VerFlip_p < random.random():
        image = image[::-1, :, :]
    if conf.random_VerFlip_show:
        cv2.imshow('1', image)
        cv2.waitKey(0)
    return image

def GussianNoise(image, conf):
    '''高斯模糊'''
    if conf.GaussNoise and conf.GaussNoise_p < random.random():
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(conf.GaussNoise_mean, conf.GaussNoise_var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        image = np.uint8(out * 255)
        if conf.GaussNoise_show:
            cv2.imshow('1', image)
            cv2.waitKey(0)
    return image

def Blur(image, conf):
    '''添加模糊'''
    if conf.blur and conf.blur_p < random.random():
        if conf.blur_size == 1:
            image = cv2.GaussianBlur(image, (11,11), 0)
        else:
            ksize = (conf.blur_size / 2) * 2 +1
            image = cv2.GaussianBlur(image, (int(ksize), int(ksize)), 0)
        if conf.blur_show:
            cv2.imshow('1', image)
            cv2.waitKey(0)
    return image

def hsv_chance(image, conf):
    '''hsv颜色变化'''
    if conf.hsv_flag and conf.hsv_p < random.random():
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hue = rand(-conf.aug_hue, conf.aug_hue)
        sat = rand(-conf.aug_sat, conf.aug_sat)
        val = rand(-conf.aug_val, conf.aug_val)
        x = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image = np.clip(cv2.cvtColor(image, cv2.COLOR_HSV2RGB), 0, 255)
        if conf.hsv_show:
            cv2.imshow('1', image)
            cv2.waitKey(0)
    return image

if __name__ == '__main__':
    from config.configV1 import ConfigV1
    phone_path = r'E:\dataset\photo2\暂时\58.jpg'
    cv_img = cv2.imdecode(np.fromfile(phone_path, dtype=np.uint8), -1)
    conf = ConfigV1()
    # get_random_size(cv_img, conf)
    # HorizonFilp(cv_img, conf)
    # VerFlip(cv_img, conf)
    # GussianNoise(cv_img, conf)
    # Blur(cv_img, conf)
    hsv_chance(cv_img, conf)