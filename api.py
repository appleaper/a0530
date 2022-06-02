from flask import Flask,request
import json
import sys
from config.configV2 import ConfigV2
from model.model_mainV2 import MultiFTNet
import os
import torch
import base64
import cv2
import numpy as np
import torch.nn.functional as F

respond_dict = {
    'code':0,       # 0为调用成功，1为失败
    'msg': {
        '0': '成功'   # 0时无报错信息，  field equired：post缺少imgBase64字符串，   Img Decode Error图片解码错误
    },
    'res':{
        'score' : 0.99,     # 置信度
        'isReshoot': False      # 是否翻拍，True为翻拍，False为非翻拍
    }
}

def base64_to_img(base64_str):
    '''传入为RGB格式下的base64，传出为RGB格式的numpy矩阵'''
    byte_data = base64.b64decode(base64_str)        # 将base64转为二进制
    encode_image = np.asarray(bytearray(byte_data), dtype='uint8')      # 二进制转换Wie一维数组
    img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)        # 用cv2解码为三通道矩阵
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)      # BGR2RGB
    return img_array

app = Flask(__name__)
@app.route('/promotion/reshoot', methods = ['POST'])
def reshoot():
    conf = ConfigV2()
    conf.test_mode = True
    conf.train_mode = False
    conf.val_mode = False
    model = MultiFTNet(conf)  # 初始化模型
    input_pick = request.get_json()
    print(input_pick)
    # 加载模型
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        model.load_state_dict(state_dict)
    else:
        respond_dict['code'] = 1
        respond_dict['msg']['0'] = 'model weight not exist'
        return json.dumps(respond_dict, indent=2, ensure_ascii=False)
    try:
        image = base64_to_img(input_pick['imgBase64'])  # base64转image
        if hasattr(image, 'shape'):
            image = image[:, :, :3]
            image = cv2.resize(image, (conf.image_w, conf.image_h))
            cv_img = image.transpose((2, 0, 1))
            cv_img = torch.from_numpy(cv_img).float()
            cv_img = torch.unsqueeze(cv_img, 0)
            cv_img = cv_img.to(conf.device)
            model.to(conf.device)
            model.eval()
            with torch.no_grad():
                output = model(cv_img)
                score = F.softmax(output, dim=1).cpu().detach().numpy()
                result = np.argmax(F.softmax(output, dim=1).cpu().detach().numpy(), 1)
            respond_dict['res']['isReshoot'] = conf.id2class[str(result[0])]
            respond_dict['res']['score'] = float(score[0][result])
            return json.dumps(respond_dict, indent=2, ensure_ascii=False)
        else:
            respond_dict['code'] = 1
            respond_dict['msg']['0'] = 'field equired'
            return json.dumps(respond_dict, indent=2, ensure_ascii=False)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    if sys.platform == 'win32':
        app.run(host='127.0.0.1', port=5454)
    else:
        app.run(host='192.168.30.51', port=5454)