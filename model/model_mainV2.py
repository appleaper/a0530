from torch import nn
from model.model_blockV2 import Conv_block,Depth_Wise, ResidualSE,Linear_block, Flatten, Linear, BatchNorm1d
import torch

class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ft(x)

class MultiFTNet(nn.Module):
    def __init__(self, conf):
        super(MultiFTNet, self).__init__()
        self.conf = conf
        self.img_channel = self.conf.channel
        self.keep = [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                      52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
        self.conv1 = Conv_block(self.img_channel, self.keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(self.keep[0], self.keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=self.keep[1])

        c1 = [(self.keep[1], self.keep[2])]
        c2 = [(self.keep[2], self.keep[3])]
        c3 = [(self.keep[3], self.keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=self.keep[3])
        c1 = [(self.keep[4], self.keep[5]), (self.keep[7], self.keep[8]), (self.keep[10], self.keep[11]),
              (self.keep[13], self.keep[14])]
        c2 = [(self.keep[5], self.keep[6]), (self.keep[8], self.keep[9]), (self.keep[11], self.keep[12]),
              (self.keep[14], self.keep[15])]
        c3 = [(self.keep[6], self.keep[7]), (self.keep[9], self.keep[10]), (self.keep[12], self.keep[13]),
              (self.keep[15], self.keep[16])]
        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=self.keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(self.keep[16], self.keep[17])]
        c2 = [(self.keep[17], self.keep[18])]
        c3 = [(self.keep[18], self.keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=self.keep[19])

        c1 = [(self.keep[19], self.keep[20]), (self.keep[22], self.keep[23]), (self.keep[25], self.keep[26]), (self.keep[28], self.keep[29]),
              (self.keep[31], self.keep[32]), (self.keep[34], self.keep[35])]
        c2 = [(self.keep[20], self.keep[21]), (self.keep[23], self.keep[24]), (self.keep[26], self.keep[27]), (self.keep[29], self.keep[30]),
              (self.keep[32], self.keep[33]), (self.keep[35], self.keep[36])]
        c3 = [(self.keep[21], self.keep[22]), (self.keep[24], self.keep[25]), (self.keep[27], self.keep[28]), (self.keep[30], self.keep[31]),
              (self.keep[33], self.keep[34]), (self.keep[36], self.keep[37])]
        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=self.keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(self.keep[37], self.keep[38])]
        c2 = [(self.keep[38], self.keep[39])]
        c3 = [(self.keep[39], self.keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=self.keep[40])

        c1 = [(self.keep[40], self.keep[41]), (self.keep[43], self.keep[44])]
        c2 = [(self.keep[41], self.keep[42]), (self.keep[44], self.keep[45])]
        c3 = [(self.keep[42], self.keep[43]), (self.keep[45], self.keep[46])]
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=self.keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(self.keep[46], self.keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(self.keep[47], self.keep[48], groups=self.keep[48], kernel=self.conf.kernel_size, stride=(1, 1),
                                      padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, self.conf.embedding, bias=False)
        self.bn = BatchNorm1d(self.conf.embedding)
        self.drop = torch.nn.Dropout(p=self.conf.drop)
        self.prob = Linear(self.conf.embedding, self.conf.num_class, bias=False)
        self.FTGenerator = FTGenerator(in_channels=128)

    def forward(self, x):       # [2,3,256,256]
        x = self.conv1(x)       # [2,32,128,128]
        x = self.conv2_dw(x)        # [2,32,128,128]
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x1 = self.conv_45(x)
        x1 = self.conv_5(x1)
        x1 = self.conv_6_sep(x1)
        x1 = self.conv_6_dw(x1)
        x1 = self.conv_6_flatten(x1)
        x1 = self.linear(x1)
        x1 = self.bn(x1)
        x1 = self.drop(x1)
        cls = self.prob(x1)
        if self.conf.train_mode:
            ft = self.FTGenerator(x)
            return cls, ft
        else:
            return cls
