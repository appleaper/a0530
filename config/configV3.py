import argparse

class ConfigV3():
    def __init__(self):
        self.cuda = self.get_args()  # 控制是否使用GPU

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=bool, default=True)
        args = parser.parse_args()
        return args

    def data_config(self):
        self.fp16 = False
        self.classes_path = 'model_data/voc_classes.txt'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]

        self.model_path = 'model_data/yolov5_s.pth'
        self.input_size = [640, 640]     # 需要是32的倍数
        self.backbone = 'cspdarknet'
        self.pretrained = False
        self.phi = 's'

        self.mosaic = True
        self.mosaic_prob = 0.5      # 每个step有多少概率使用mosaic数据增强，默认50%。
        self.mixup = True       # 是否使用mixup数据增强，仅在mosaic=True时有效。只会对mosaic增强后的图片进行mixup的处理。
        self.mixup_prob = 0.5   # 有多少概率在mosaic后使用mixup数据增强，默认50%。总的mixup概率为mosaic_prob * mixup_prob。

        '''
        参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
        当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。默认为前70%个epoch，100个世代会开启70个世代。
        '''
        self.special_aug_ratio = 0.7
        self.label_smoothing = 0        # label_smoothing     标签平滑。一般0.01以下。如0.01、0.005。

        self.Init_Epoch = 0
        self.Freeze_Epoch = 50
        self.Freeze_batch_size = 16

        self.Unfreeze_Epoch = 300
        self.Unfreeze_batch_size = 8

        self.Freeze_Train = True    # 是否进行冻结训练，默认先冻结主干后解冻训练
        self.Init_lr = 1e-2     # 模型的最大学习率
        self.Min_lr = self.Init_lr * 0.01   # 模型的最小学习率

        self.optimizer_type = 'sgd'     # Adam时建议lr=1e-3 SGD时建议lr=1e-2
        self.momentum = 0.937
        self.weight_decay = 5e-4    # 权值衰减，可防止过拟合, adam会导致weight_decay错误，使用adam时建议设置为0。

        self.lr_decay_type = 'cos'      # 学习率下降方式， 可选方式：step、cos
        self.save_period = 10           # 多少个epoch保存一次权重
        self.save_dir = 'logs'          # 权值和日志保存的文件夹

        self.eval_flag = True       # 是否在训练时进行评估，评估对象是验证集
        self.eval_period = 10       # 代表多少个epoch评估一次，评估比较耗时，频繁评估会导致训练很慢
        self.num_workers = 4        # 设置是否使用多线程读取数据，开启后占用更多内存

        self.train_annotation_path = '2007_train.txt'   # 训练图片路径和标签
        self.val_annotation_path = '2007_val.txt'       # 验证图片路径和标签

        