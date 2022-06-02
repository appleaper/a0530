import requests
import base64
import os
import time
import json
import cv2
import sys

def img2base64(img_array):
    '''传入图片为RGB格式numpy矩阵，传出的base64也通过RGB的编码'''
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)      # RGB2BGR 用于cv2编码
    encode_image = cv2.imencode('.jpg', img_array)[1]       # 用cv2压缩/编码 转为一维数组
    byte_data = encode_image.tobytes()      # 转换为二进制
    base64_str = base64.b64encode(byte_data).decode('ascii')        # 转换为base64
    return base64_str

if sys.platform.startswith('linux'):
    test_url = 'http://192.168.30.51:5454/promotion/reshoot'
else:
    test_url = 'http://127.0.0.1:5454/promotion/reshoot'

imgpath = "./demo/6.png"
with open(imgpath,'rb') as f:
    img = base64.b64encode(f.read()).decode()

# img_array = cv2.imread(imgpath)
# img = img2base64(img_array)

headers = {'Content-Type': 'application/json'}

res = {
    "taskid":"1",
    "imgBase64": img
}

json_text = json.dumps(res, indent=2, ensure_ascii=False)

response = requests.request("POST", test_url, headers=headers, data=json_text)
print(response.text)