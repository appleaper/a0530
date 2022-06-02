import os
import numpy as np

def read_txt(path, lines, mode = 'w'):
    with open(path, mode, encoding='utf8') as f:
        for line in lines:
            f.writelines(line)

def demo1():
    train_dir = r'E:\dataset\MR-GAN\train'
    lines = []
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if class_name == 'fake':
                label = '翻拍'
            else:
                label = '非翻拍'
            lines.append(file_path + '\t' + label + '\n')

    test_dir = r'E:\dataset\MR-GAN\test'
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if class_name == 'fake':
                label = '翻拍'
            else:
                label = '非翻拍'
            lines.append(file_path + '\t' + label + '\n')

    num = len(lines)
    train_num = int(0.6 * num)
    val_num = int(0.2 * num)
    np.random.shuffle(lines)
    train_data = lines[:train_num]
    val_data = lines[train_num : train_num+val_num]
    test_data = lines[train_num+val_num:]
    read_txt('./train.txt', train_data)
    read_txt('./val.txt', val_data)
    read_txt('./test.txt', test_data)

def demo2():
    fake1_path = r'E:\dataset\MR-GAN\train\fake1'
    unfake1_path = r'E:\dataset\MR-GAN\train\unfake1'
    fake2_path = r'E:\dataset\MR-GAN\test\fake1'

    lines = []
    for file_name in os.listdir(fake1_path):
        path = os.path.join(fake1_path, file_name)
        lines.append(str(path) + '\t' + '翻拍' + '\n')
    read_txt('./test1.txt',lines, 'w')

    lines = []
    for file_name in os.listdir(fake2_path):
        path = os.path.join(fake2_path, file_name)
        lines.append(str(path) + '\t' + '翻拍' + '\n')
    read_txt('./test1.txt', lines, 'a')

    lines = []
    for file_name in os.listdir(unfake1_path):
        path = os.path.join(unfake1_path, file_name)
        lines.append(str(path) + '\t' + '非翻拍' + '\n')
    read_txt('./test1.txt', lines, 'a')

def read_txt_data(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    return lines

def demo3():
    test_data = './test.txt'
    # test_data1 = './test1.txt'
    train_data = './train.txt'
    val_data = './val.txt'
    lines = read_txt_data(test_data)
    # lines1 = read_txt_data(test_data1)
    lines2 = read_txt_data(train_data)
    lines3 = read_txt_data(val_data)
    # lines.extend(lines1)
    lines.extend(lines2)
    lines.extend(lines3)
    new_lines = []
    for line in lines:
        if len(line) == 0:
            continue
        new_lines.append(line)
    np.random.shuffle(new_lines)
    total = len(new_lines)
    train_num = int(0.6 * total)
    val_num = int(0.2 * total)
    read_txt('./train.txt', new_lines[:train_num], 'w')
    read_txt('./val.txt', new_lines[train_num: train_num+val_num], 'w')
    read_txt('./test.txt', new_lines[train_num+val_num:], 'w')

def demo4():
    line1 = read_txt_data('./test.txt')
    line2 = read_txt_data('./train.txt')
    line3 = read_txt_data('./val.txt')
    line1.extend(line2)
    line1.extend(line3)
    read_txt('./all.txt', line1, 'w')

def demo5():
    image_dir = r'E:\dataset\VOC2012\VOCdevkit\JPEGImages'
    num = 0
    total = 4895
    lines = []
    for file_name in os.listdir(image_dir):
        path = os.path.join(image_dir, file_name)
        label = '非翻拍'
        lines.append([str(path) + '\t' + label + '\n'])
        total -= 1
        if total==0:
            break
    read_txt('./all.txt', lines, 'a')

if __name__ == '__main__':
    new_lines = read_txt_data('./all.txt')
    np.random.shuffle(new_lines)
    total = len(new_lines)
    train_num = int(0.6 * total)
    val_num = int(0.2 * total)
    read_txt('./train.txt', new_lines[:train_num], 'w')
    read_txt('./val.txt', new_lines[train_num: train_num+val_num], 'w')
    read_txt('./test.txt', new_lines[train_num+val_num:], 'w')