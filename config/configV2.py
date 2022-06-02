import argparse
from util.tool import read_txt
import torch

class ConfigV2():
    def __init__(self):
        self.cuda = self.get_args()     # 控制是否使用GPU
        self.data_config()              # 数据有关的配置
        self.model_config()             # 模型结构配置
        self.aug_config()               # 数据增强配置
        self.train_config()             # 训练配置
        self.device_config()            # 设备配置

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=bool, default=True)
        args = parser.parse_args()
        return args

    def data_config(self):
        self.train_dataset_path = r'E:\software\inspur\0530\data\train.txt'     # 训练数据集，需要自己生成
        self.val_dataset_path = r'E:\software\inspur\0530\data\val.txt'
        self.test_dataset_path = r'E:\software\inspur\0530\data\test.txt'
        self.train_mode = True      # 训练模式
        self.val_mode = True        # 验证模式
        self.test_mode = False
        self.class_name_list = ['翻拍', '非翻拍']        # 类别名字

        if self.train_mode:
            self.train_data_list = read_txt(self.train_dataset_path)        # 读取所有文件列表
        if self.val_mode:
            self.val_data_list = read_txt(self.val_dataset_path)
        if self.test_mode:
            self.test_data_list = read_txt(self.test_dataset_path)

        self.class2id = {}      # 类别转id
        self.id2class = {}      # id2类别
        for index, class_name in enumerate(self.class_name_list):
            index_str = str(index)
            self.class2id[class_name] = index_str
            self.id2class[index_str] = class_name
        self.num_class = len(self.class_name_list)      # 类别数目

    def model_config(self):
        self.image_h = 256      # 图片高度
        self.image_w = 256      # 图片宽度
        self.channel = 3        # 图片通道
        self.embedding = 512    # 向量数量
        self.drop = 0.8         # 丢弃概率
        self.kernel_size = self.get_kernel(self.image_h, self.image_w)      # 与模型结构有关

    def get_kernel(self, height, width):
        kernel_size = ((height + 15) // 16, (width + 15) // 16)
        return kernel_size

    def aug_config(self):
        self.train_aug = True       # 控制是否数据加强
        self.ft_height = 2 * self.kernel_size[0]    # 查看ft高度
        self.ft_width = 2 * self.kernel_size[1]     # 查看ft宽度

    def train_config(self):
        self.start_epoch = 1        # 开始epoch
        self.end_epoch = 100        # 结束epoch
        self.lr = 1e-1              # 学习率
        self.momentum = 0.9         # 优化器参数
        self.weight_decay = 5e-4    # 权值衰减，可以防止过拟合
        self.gamma = 0.1
        millor1 = int(self.end_epoch * 0.4)
        millor2 = int(self.end_epoch * 0.6)
        millor3 = int(self.end_epoch * 0.8)
        self.milestones = [millor1, millor2, millor3]       # 里程碑
        self.batch_size = 32            # 批量大小
        self.save_path = r'./save_dir'      # 保存路径
        self.pre_train_path = r'./save_dir/fanpaiSE_94.pth'      # 预训练权重
        self.log_dir = r'./log'          # 日志路径
        self.model_name = 'fanpaiV1'        # 模型名字

    def device_config(self):
        if self.cuda:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'

        if self.device == 'cuda':
            self.device = torch.device(self.device)     # 设备