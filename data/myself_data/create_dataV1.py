import os
import config.configV1 as conf_temp
import random

def write_line(lines, file_name):
    with open(file_name, 'w', encoding='utf8') as f:
        for line in lines:
            f.writelines(line)

def get_file_list(dir_path):
    lines = []
    for class_name in os.listdir(dir_path):
        path1 = os.path.join(dir_path, class_name)
        for file_name in os.listdir(path1):
            image_path = os.path.join(path1, file_name)
            out_str = str(image_path) + '\t' + class_name + '\n'
            lines.append(out_str)
    return lines

def demo1():
    conf = conf_temp.ConfigV1()
    class_name = conf.class_name_list
    dir_path = r'E:\dataset\photo'
    lines = get_file_list(dir_path)
    dir_path1 = r'E:\dataset\photo3'
    lines1 = get_file_list(dir_path1)
    lines.extend(lines1)
    random.shuffle(lines)
    train_num = int(0.8 * len(lines))
    val_num = int(0.1 * len(lines))
    test_num = len(lines) - train_num - val_num
    write_line(lines, 'all.txt')
    write_line(lines[:train_num], 'train.txt')
    write_line(lines[train_num:train_num+val_num], 'val.txt')
    write_line(lines[train_num+val_num:], 'test.txt')

if __name__ == '__main__':
    dir_name = r'E:\dataset\photo2\1'
    lines = []
    for filename in os.listdir(dir_name):
        path = os.path.join(dir_name, filename)
        label = '插入'
        lines.append([str(path) + '\t' + label + '\n'])
    write_line(lines, './test1.txt')