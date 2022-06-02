from torch.utils.data.dataset import Dataset
from augment.transformV1 import *
import torch
from tqdm import tqdm

class TrainFanpaiV1(Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.get_train_data()

    def get_train_data(self):
        self.train_dict = {}
        for index, line in enumerate(self.conf.train_data_list):
            line = line.replace('\n','')
            train_path, label = line.split('\t')
            self.train_dict[index] = {}
            self.train_dict[index]['path'] = train_path
            self.train_dict[index]['label'] = self.conf.class2id[label]

    def __len__(self):
        return len(self.train_dict)

    def __getitem__(self, index):
        path = self.train_dict[index]['path']
        label = self.train_dict[index]['label']
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if self.conf.train_aug:
            cv_img = get_random_size(cv_img, self.conf)
            cv_img = HorizonFilp(cv_img, self.conf)
            cv_img = VerFlip(cv_img, self.conf)
            cv_img = GussianNoise(cv_img, self.conf)
            cv_img = Blur(cv_img, self.conf)
            cv_img = hsv_chance(cv_img, self.conf)
        cv_img = cv2.resize(cv_img, (self.conf.image_w, self.conf.image_h))
        cv_img = cv_img.transpose((2,0,1))
        cv_img = torch.from_numpy(cv_img).float()
        label = int(label)
        return cv_img, label

class ValFanpaiV1(Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.get_val_data()

    def get_val_data(self):
        self.val_dict = {}
        for index, line in enumerate(self.conf.val_data_list):
            line = line.replace('\n', '')
            val_path, label = line.split('\t')
            self.val_dict[index] = {}
            self.val_dict[index]['path'] = val_path
            self.val_dict[index]['label'] = self.conf.class2id[label]

    def __len__(self):
        return len(self.val_dict)

    def __getitem__(self, index):
        path = self.val_dict[index]['path']
        label = self.val_dict[index]['label']
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        cv_img = cv2.resize(cv_img, (self.conf.image_w, self.conf.image_h))
        cv_img = cv_img.transpose((2, 0, 1))
        cv_img = torch.from_numpy(cv_img).float()
        label = int(label)
        return cv_img, label

if __name__ == '__main__':
    from config.configV1 import ConfigV1
    config_temp = ConfigV1()
    val_class = ValFanpaiV1(config_temp)
    for image, label in tqdm(val_class):
        pass