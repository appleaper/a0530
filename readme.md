### 暂时拥有的功能

图片分类，翻拍



### 文件介绍

```
choise.py: 选择模型
pred.py： 单例测试
api.py： 单例接口
api_test.py: 测试接口是否正常
train.py： 具体的模型训练过程
requirement.txt: 依赖包
readme.md:	说明文档
```



### 数据集制作

图片分类和翻拍数据方式相同，制作train.txt，test.txt，val.txt

train.txt

```
E:\dataset\MR-GAN\train\fake\1508.png	翻拍
E:\dataset\MR-GAN\train\fake\3381.png	翻拍
E:\dataset\MR-GAN\train\unfake\5337.png	非翻拍
```

train.txt和val.txt，test.txt格式相同，都是path+' \ t ' + label+' \ n'



### 接口使用

运行该指令，让服务启动

```
python api.py
```

注意事项：

1. 权重位置在config/configV2.py/self.pre_train_path
2. 输入的base64图片的

输入：

```
res={
	"imgBase64" : base64_img
}
```

输出：

```
respond_dict = {
    'code':0,    
    'msg': {
        '0': '成功'
    },
    'res':{
        'score' : 0.99,    
        'isReshoot': False 
    }
}
```

code值中0为调用成功，1为失败

msg中0中的值为“成功”，则无报错信息，若0的值为field required则代表post缺少imgBase64字符串， 

score代表模型输出的置信度

isReshoot代表是否翻拍，True为翻拍，False为非翻拍



### 测试报告

测试图片来源：图片来自于网络

测试集是由数据集随机划分而来，按8:1:1比例进行划分。

数据集总数据量：49668张

| 项目           | 数量  |
| -------------- | ----- |
| 训练集         | 39734 |
| 测试集         | 4968  |
| 验证集         | 4968  |
| 总数量         | 49688 |
| 翻拍图片数量   | 24834 |
| 非翻拍图片数量 | 24834 |

测试集的准确率 = 95.2%
