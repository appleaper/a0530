import datetime
import logging
import os
import sys
import numpy as np
import random
import torch

def init_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_logger(log_file=None, log_dir=None, mode='w', stdout=True,log_level=logging.INFO,name=None):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/logs/'
    if log_file is None:
        log_file = 'log_' +str(name)+'_'+ get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    init_seed(123)
    return logging