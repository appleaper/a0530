
使用步骤
```python
pip install pytorch-pretrained-bert pytorch-nlp
```

TensorFlow和torch不能同时安装的问题一直存在，不知道怎么解决
1.创建了一个conda环境
```python
conda create –n tensorflow2-gpu python=3.7
```
2.直接安装包
```python
pip install tensorflow-gpu==2.2.0
```
3.安装cuda10.1的环境
conda install cudatoolkit=10.1 cudnn=7.6.5

1.安装适合的cuda10.1的torch，做到同一个环境兼容tensorflow和pytorch
```python
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

安装新的东西
```python
pip install pytorch-transformers
pip install transformers
```

安装pandas
```python
pip install pandas
```

安装预训练bert
```python
pip install pytorch-pretrained-bert
```

matplotlib安装
```python
pip install matplotlib
```

scikit-learn
```python
pip install scikit-learn
```

keras安装
```python
pip install keras
```

表格读取
```python
pip install openpyxl
```

jupyter notebook 切换环境
```python
pip install ipykernel
```
将环境注入到jupyter notebook中
```python
python -m ipykernel install --user --name luo --display-name "Python [conda env:luo]"
```
luo替换为已经创建的环境的名字
```python
python -m ipykernel install --user --name tensorflow2-gpu --display-name "Python [conda env:tensorflow2-gpu]"
```
遇到jupyter notebook打不开的问题
```python
pip install jupyter notebook
```

安装与视觉相关的
```python
pip install ffmpeg-python==0.2.0
```

对匈牙利算法的实现
```python
pip install munkres==1.1.2
```

安装opencv
```python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python==3.4.17.63
```

安装图形库
```python
pip install Pillow==5.4
```
安装的时候遇到报错了：

ERROR: Could not install packages due to an OSError: [WinError 5] 拒绝访问。: 'e:\\software\\conda\\envs\\tensorflow2-gpu\\lib\\site-packages\\~il\\msvcp140.dll'
Consider using the `--user` option or check the permissions.
这个错误发改的意思是权限不够，解决办法：
python -m pip install visualdl

vidgear主要是配合opencv进行视频读取更快，主要功能是对视频的读写、处理、发送、接收等
```python
pip install vidgear==0.1.4
```

安装mmcv
```python
pip install mmcv
```

安装与C++有关的包
```python
pip install editdistance
```

处理2D多边形
```python
pip install Polygon3
```

多边形裁剪
```python
pip install pyclipper
```

python和C混合编程
```python
pip install Cython
```

安装TensorboardX
```python
pip install tensorboardX
```

安装scikit-image

```
pip install scikit-image
```

FFmpeg-python

```
pip install ffmpeg-python
```

视频安装库

```
pip install gluoncv
pip install rarfile
```

python可视化
```python
pip install seaborn==0.11.0
```

数据增强加强包
```python
pip install imgaug
```

7z解压
```python
pip install py7zr
```

图片mask调色板

```
pip install imgviz
```

slowfast环境准备

fvcore
```python
pip install fvcore
```

```python
pip install simplejson
```

EasyDict

```
pip install easydict
```

数据增强库

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple albumentations
```

数据增强

```
pip install imgaug
```

导出项目依赖包

```
pip install pipreqs
```

