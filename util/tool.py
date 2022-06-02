import os
import pickle as pkl
import numpy as np

def make_if_not_exist(folder_path):
    '''创建目录'''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_data(passage, path):
    '''
    保存pickple数据
    :param passage: 要保存的内容
    :param path: 保存的路径
    :return:
    '''
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    f = open(path, 'wb')
    pkl.dump(passage, f)
    f.close()

def load_data(path):
    '''
    加载pickple文件
    :param path: 文件的路径
    :return: 文件的内容
    '''
    if os.path.exists(path):
        with open(path, 'rb') as f:
            all_dict = pkl.load(f)
        return all_dict
    else:
        assert False, 'file not exist'

def random_num():
    '''随机名字'''
    small = ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']
    big = ['Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
    num = ['1','2','3','4','5','6','7','8','9','0']

    num1 = 0
    out_str = ''
    while True:
        a = np.random.choice(small)
        b = np.random.choice(big)
        c = np.random.choice(num)
        out_str += a+b+c
        num1 += 1
        if num1>2:
            break
    return out_str

def rename(dir_path):
    '''文件改名'''
    for file_name in os.listdir(dir_path):
        src_path = os.path.join(dir_path, file_name)
        other_name = random_num()
        dst_path = os.path.join(dir_path, other_name+'.jpg')
        os.rename(src_path, dst_path)

def read_txt(path):
    '''读取文件'''
    lines = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    return lines

if __name__ == '__main__':
    pass