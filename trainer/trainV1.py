from torch.nn import CrossEntropyLoss, MSELoss
from model.model_mainV1 import Resnet18,Resnet50,mobilenet_v2,vgg16
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import os
from model.model_skip import weight_init

class TrainMainV1():
    def __init__(self, conf, train_loader, val_loader, logging):
        self.conf = conf
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logging = logging
        self.train_model()

    def train_model(self):
        self.cls_criterion = CrossEntropyLoss()

        self.model = Resnet18(self.conf)
        # self.model = mobilenet_v2(self.conf)
        # self.model = vgg16(self.conf)

        self.model.to(self.conf.device)
        if not os.path.exists(self.conf.pre_train_path):
            weight_init(self.model)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=self.conf.weight_decay,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

    def _train_stage(self):
        self.acc = 0
        self.model.train()
        for epoch in range(self.conf.start_epoch, self.conf.end_epoch):
            pbar = tqdm(iter(self.train_loader), total=len(self.train_loader))
            pbar.set_description(f'epoch[{epoch}/{self.conf.end_epoch}]')
            total = 0
            true_num = 0
            self.model.train()
            for sample, target in pbar:
                self.optimizer.zero_grad()
                sample = sample.to(self.conf.device)
                target = target.to(self.conf.device)
                output = self.model.forward(sample)
                loss = self.cls_criterion(output, target)
                result = np.argmax(F.softmax(output, dim=1).cpu().detach().numpy(), 1)
                target_cpu = target.cpu().detach().numpy()
                true_num += (result == target_cpu).sum()
                total += sample.shape[0]
                acc = true_num / total
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(acc=acc)
            self.schedule_lr.step()
            torch.cuda.empty_cache()
            if self.conf.val_mode:
                acc = self.val(self.model)
                if self.acc < acc:
                    self.acc = acc
                    torch.save(self.model.state_dict(), os.path.join(self.conf.save_path, f'{self.conf.model_name}' + "_" + f'{int(acc * 100)}.pth'))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.conf.save_path, f'{self.conf.model_name}' + "_" + f'{str(epoch)}.pth'))

    def val(self, model):
        total = 0
        right = 0
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
        return acc