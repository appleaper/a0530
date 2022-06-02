import cv2
from config.configV2 import ConfigV2
import torch
import numpy as np
from util.tool import read_txt
import os
from model.model_mainV2 import MultiFTNet
import torch.nn.functional as F
from tqdm import tqdm
import base64

def img2base64(img_array):
    '''传入图片为RGB格式numpy矩阵，传出的base64也通过RGB的编码'''
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)      # RGB2BGR 用于cv2编码
    encode_image = cv2.imencode('.jpg', img_array)[1]       # 用cv2压缩/编码 转为一维数组
    byte_data = encode_image.tobytes()      # 转换为二进制
    base64_str = base64.b64encode(byte_data).decode('ascii')        # 转换为base64
    return base64_str

def base64_to_img(base64_str):
    '''传入为RGB格式下的base64，传出为RGB格式的numpy矩阵'''
    byte_data = base64.b64decode(base64_str)        # 将base64转为二进制
    encode_image = np.asarray(bytearray(byte_data), dtype='uint8')      # 二进制转换Wie一维数组
    img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)        # 用cv2解码为三通道矩阵
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)      # BGR2RGB
    return img_array

if __name__ == '__main__':
    conf = ConfigV2()
    conf.test_mode = True
    conf.train_mode = False
    conf.val_mode = False
    lines = read_txt('./data/test.txt')
    model = MultiFTNet(conf)        # 初始化模型
    # 加载模型
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        model.load_state_dict(state_dict)

    total = 0
    true_flag = 0
    pbar = tqdm(lines, total=len(lines))
    for line in pbar:
        line = line.replace('\n', '')
        path, label = line.split('\t')
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        image64 = img2base64(image)     # 转base64
        image = base64_to_img(image64)      # base64转image
        image = image[:, : , :3]
        try:
            image.shape
        except:
            print('can not read the image')
            continue
        image = cv2.resize(image, (conf.image_w, conf.image_h))
        cv_img = image.transpose((2, 0, 1))
        cv_img = torch.from_numpy(cv_img).float()
        cv_img = torch.unsqueeze(cv_img, 0)
        cv_img = cv_img.to(conf.device)
        model.to(conf.device)
        model.eval()

        # 模型推理
        with torch.no_grad():
            output = model(cv_img)
            result = np.argmax(F.softmax(output, dim=1).cpu().detach().numpy(), 1)
            if conf.id2class[str(result[0])]==label:
                true_flag += 1
            total += 1
        pbar.set_postfix(acc = true_flag / total)