from torch.nn import CrossEntropyLoss, MSELoss
from model.model_mainV2 import MultiFTNet
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import os

class TrainMainV2():
    def __init__(self, conf, train_loader, val_loader, logging):
        self.conf = conf        # 配置
        self.train_loader = train_loader    # 训练数据
        self.val_loader = val_loader        # 验证数据
        self.logging = logging
        self.train_model()

    def train_model(self):
        self.cls_criterion = CrossEntropyLoss()     # 分类损失
        self.mse_criterion = MSELoss()              # 回归损失

        self.logging.info('init model')
        self.model = MultiFTNet(self.conf)          # 初始化模型
        # 加载预训练模型
        if os.path.exists(self.conf.pre_train_path):
            state_dict = torch.load(self.conf.pre_train_path, map_location=self.conf.device)
            self.model.load_state_dict(state_dict)
        self.model.to(self.conf.device)

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=self.conf.weight_decay,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

    def _train_stage(self):
        self.acc = 0
        for epoch in range(self.conf.start_epoch, self.conf.end_epoch):
            pbar = tqdm(iter(self.train_loader), total=len(self.train_loader))
            pbar.set_description(f'epoch[{epoch}/{self.conf.end_epoch}]')
            total = 0
            true_num = 0
            self.model.train()
            self.conf.train_mode = True
            for sample, ft_sample, target in pbar:
                self.optimizer.zero_grad()
                target = target.to(self.conf.device)
                sample = sample.to(self.conf.device)
                ft_sample = ft_sample.to(self.conf.device)
                output, feature = self.model.forward(sample)
                loss_cls = self.cls_criterion(output, target)
                result = np.argmax(F.softmax(output, dim=1).cpu().detach().numpy(), 1)
                target_cpu = target.cpu().detach().numpy()
                true_num += (result == target_cpu).sum()
                total += sample.shape[0]
                acc = true_num / total

                loss_fea = self.mse_criterion(feature, ft_sample.to(self.conf.device))
                loss = 0.5 * loss_cls + 0.5 * loss_fea
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(acc = acc)
            self.logging.info(f'train_mode: [{epoch}/{self.conf.end_epoch}] acc:{acc}')
            self.schedule_lr.step()
            torch.cuda.empty_cache()
            if self.conf.val_mode:
                acc = self.val(self.model)
                if self.acc < acc:
                    self.acc = acc
                    torch.save(self.model.state_dict(), os.path.join(self.conf.save_path,
                                                                     f'{self.conf.model_name}' + "_" + f'{int(acc * 100)}.pth'))
            else:
                torch.save(self.model.state_dict(),
                           os.path.join(self.conf.save_path, f'{self.conf.model_name}' + "_" + f'{str(epoch)}.pth'))

    def val(self, model):
        total = 0
        right = 0
        self.conf.train_mode = False
        pbar = tqdm(iter(self.val_loader), total=len(self.val_loader))
        for sample, target in pbar:
            sample = sample.to(self.conf.device)
            model.eval()
            with torch.no_grad():
                output = model(sample)
                output = np.argmax(F.softmax(output, dim=1).cpu().detach().numpy(), 1)
                target = target.numpy()
                right += (output == target).sum()
                total += output.shape[0]
                pbar.set_postfix(acc=right/total)

        torch.cuda.empty_cache()
        acc = right / total
        self.logging.info(f'test_mode: acc:{acc}')
        return acc
