import argparse
import torch
from util.tool import read_txt

class ConfigV1():
    def __init__(self):
        self.cuda = self.get_args()
        self.model_config()
        self.train_config()
        self.data_config()
        self.aug_config()
        self.device_config()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=bool, default=True)
        args = parser.parse_args()
        return args

    def model_config(self):
        self.image_h = 256
        self.image_w = 256
        self.channel = 3
        self.embedding = 512
        self.drop = 0.8
        self.kernel_size = self.get_kernel(self.image_h, self.image_w)

    def train_config(self):
        self.start_epoch = 1
        self.end_epoch = 100
        self.lr = 1e-1
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.gamma = 0.1
        self.download = False
        millor1 = int(self.end_epoch * 0.4)
        millor2 = int(self.end_epoch * 0.6)
        millor3 = int(self.end_epoch * 0.8)
        self.milestones = [millor1, millor2, millor3]
        self.batch_size = 128        # mobilev2:32  resnet18:128
        self.save_path = r'./save_dir'
        # self.pre_train_path = r'./pre_weight/mobilenet_v2.pth'
        self.pre_train_path = './temp/myself_95.pth'
        self.log_dir = r'./log'
        self.model_name = 'fanpai'

    def data_config(self):
        self.train_dataset_path = r'E:\software\inspur\0530\data\myself_data\train.txt'
        self.val_dataset_path = r'E:\software\inspur\0530\data\myself_data\val.txt'
        self.test_dataset_path = r'E:\software\inspur\0530\data\myself_data\test.txt'
        self.train_mode = True
        self.val_mode = True
        self.test_mode = False
        self.class_name_list = []

        if self.train_mode:
            self.train_data_list = read_txt(self.train_dataset_path)
        if self.val_mode:
            self.val_data_list = read_txt(self.val_dataset_path)
        if self.test_mode:
            self.test_data_list = read_txt(self.test_dataset_path)

        self.class2id = {}
        self.id2class = {}
        for index, class_name in enumerate(self.class_name_list):
            index_str = str(index)
            self.class2id[class_name] = index_str
            self.id2class[index_str] = class_name
        self.num_class = len(self.class_name_list)

    def aug_config(self):
        self.train_aug = True

        # 缩放有关
        self.random_scale = True
        self.random_scale_p = 0.5
        self.random_scale_show = False
        self.jitter = 0.3
        self.scale_min = 0.25
        self.scale_max = 2

        # 左右翻转
        self.random_HorFilp = True
        self.random_HorFilp_p = 0.5
        self.random_HorFilp_show = False

        # 上下翻转
        self.random_VerFlip = True
        self.random_VerFlip_p = 0.5
        self.random_VerFlip_show = False

        # 高斯噪声
        self.GaussNoise = True
        self.GaussNoise_p = 0.5
        self.GaussNoise_mean = 0
        self.GaussNoise_var = 0.001
        self.GaussNoise_show = False

        # 模糊
        self.blur = True
        self.blur_p = 0.5
        self.blur_show = False
        self.blur_size = 1      # 1明显模糊，2是小模糊

        # hsv变化
        self.hsv_flag = True
        self.hsv_p = 0.5
        self.hsv_show = False
        self.aug_hue = 0.1
        self.aug_sat = 1.5
        self.aug_val = 1.5

    def get_kernel(self, height, width):
        kernel_size = ((height + 15) // 16, (width + 15) // 16)
        return kernel_size

    def device_config(self):
        if self.cuda:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

        if self.device == 'cuda':
            self.device = torch.device(self.device)
