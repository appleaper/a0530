from torch.utils.data.dataset import Dataset
from augment.transformV2 import *
import warnings

def generate_FT(image):
    '''傅里叶变化'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg

class TrainFanpaiV2(Dataset):
    def __init__(self, conf):
        super(TrainFanpaiV2, self).__init__()
        self.conf = conf
        self.get_train_data()
        self.train_transform = Compose([
                ToPILImage(),
                RandomResizedCrop(size=(self.conf.image_h, self.conf.image_w),
                                  scale=(0.9, 1.1)),        # 随机裁切
                ColorJitter(brightness=0.4,
                            contrast=0.4, saturation=0.4, hue=0.1),     # 颜色抖动
                RandomRotation(10),     # 旋转
                RandomHorizontalFlip(),     # 翻转
                ToTensor()      # 转tensor
            ])      # 增强措施

    def get_train_data(self):
        '''读取训练的文件列表'''
        self.train_dict = {}
        for index, line in enumerate(self.conf.train_data_list):
            line = line.replace('\n','')
            train_path, label = line.split('\t')
            self.train_dict[index] = {}
            self.train_dict[index]['path'] = train_path
            self.train_dict[index]['label'] = label

    def __len__(self):
        return len(self.train_dict)

    def __getitem__(self, index):
        path = self.train_dict[index]['path']       # 文件路径
        label = self.conf.class2id[self.train_dict[index]['label']]     # 图片对应的标签
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)      # 读取图片

        # 防止读取不了
        if not hasattr(cv_img, 'shape'):
            warnings.warn(f'can not load {path}')
            try:
                index = min(0, index - 1)
                path = self.train_dict[index]['path']
                cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            except:
                assert False, f'can not load {path}'
        ft_sample = generate_FT(cv_img)     # 傅里叶变化
        ft_sample = cv2.resize(ft_sample, (self.conf.ft_width, self.conf.ft_height))    # 控制形状
        ft_sample = torch.from_numpy(ft_sample).float()     # 转tensor
        ft_sample = torch.unsqueeze(ft_sample, 0)       # 加一个维度
        if self.conf.train_aug:
            cv_img = self.train_transform(cv_img)
            cv_img = cv_img[:3, :, :]       # 防止读取的png的4个通道
        label = int(label)
        return cv_img, ft_sample, label

class ValFanpaiV2(Dataset):
    '''与训练的数据处理类似'''
    def __init__(self, conf):
        super(ValFanpaiV2, self).__init__()
        self.conf = conf
        self.get_val_data()

    def get_val_data(self):
        self.val_dict = {}
        for index, line in enumerate(self.conf.val_data_list):
            line = line.replace('\n','')
            val_path, label = line.split('\t')
            self.val_dict[index] = {}
            self.val_dict[index]['path'] = val_path
            self.val_dict[index]['label'] = label

    def __len__(self):
        return len(self.val_dict)

    def __getitem__(self, index):
        path = self.val_dict[index]['path']
        label = self.conf.class2id[self.val_dict[index]['label']]
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if not hasattr(cv_img, 'shape'):
            warnings.warn(f'can not load {path}')
            try:
                index = min(0, index - 1)
                path = self.val_dict[index]['path']
                cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            except:
                assert False, f'can not load {path}'
        cv_img = cv_img[:3, :, :]
        label = int(label)
        return cv_img, label

if __name__ == '__main__':
    from config.configV2 import ConfigV2
    from tqdm import tqdm
    config_temp = ConfigV2()
    train_class = TrainFanpaiV2(config_temp)
    for image, ft_sample, label in tqdm(train_class):
        pass