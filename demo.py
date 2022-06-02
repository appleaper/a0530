# -*- coding:utf-8 -*-

import subprocess
import argparse
import os
import numpy as np


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

input_video = r'D:\video1'
output_video = r'D:\video_class'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time', '-s', type=str, default='')
    parser.add_argument('--end_time', '-e', type=str, default='')
    parser.add_argument('--file_name', '-f', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    input_file = os.path.join(input_video, args.file_name + '.mp4')
    out_str = random_num()
    output_file = os.path.join(output_video, out_str + '.mp4')
    # ffmpeg -i Screenrecorder-2021-07-20-00-08-39-748.mp4 -ss 00:00:08 -to 00:00:19 Screenrecorder-2021-07-20-00-08-39-748.mp4
    subprocess.call((f"ffmpeg -i {input_file} -ss 00:{args.start_time} -to 00:{args.end_time} {output_file}"),shell=True)