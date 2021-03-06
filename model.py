import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import global_avgpool2d, winner_take_all


class FBSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, fbs=False, sparsity_ratio=1.0,
                 test=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fbs = fbs
        self.sparsity_ratio = sparsity_ratio
        self.test = test

        if fbs:
            self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels))

            self.bn.weight.requires_grad = False
            nn.init.kaiming_uniform_(self.weights, mode='fan_out', nonlinearity='relu')
            nn.init.ones_(self.bias)

    def forward(self, x):
        if self.fbs:
            return self.fbs_forward(x)
        else:
            return self.original_forward(x)

    def original_forward(self, x):
        if not self.test:
            x = self.conv(x)
            x = self.bn(x)
            x = F.relu(x)
            return x
        else:
            in_channels = (x.abs().sum(dim=(2, 3)) > 1e-15).sum(dim=-1)
            x = self.conv(x)
            out_channels = (x.abs().sum(dim=(2, 3)) > 1e-15).sum(dim=-1)
            H, W = x.size(2), x.size(3)

            MACs = (self.kernel_size ** 2) * in_channels * out_channels * H * W
            x = self.bn(x)
            x = F.relu(x)
            return x, MACs

    def fbs_forward(self, x):
        in_channels = (x.abs().sum(dim=(2, 3)) > 1e-15).sum(dim=-1)
        ss = global_avgpool2d(x)
        g = F.relu(F.linear(ss, self.weights, self.bias))

        g_wta, winner_mask = winner_take_all(g, self.sparsity_ratio)

        x = self.conv(x)

        if not self.training:
            x = x * winner_mask.unsqueeze(2).unsqueeze(3)

        out_channels = (x.abs().sum(dim=(2, 3)) > 1e-15).sum(dim=-1)
        H, W = x.size(2), x.size(3)

        MACs = (self.kernel_size ** 2) * in_channels * out_channels * H * W

        x = self.bn(x)
        x = x * g_wta.unsqueeze(2).unsqueeze(3)
        x = F.relu(x)

        if self.test:
            return x, g.abs().sum(dim=1).mean(), MACs
        else:
            return x, g.abs().sum(dim=1).mean()


class CifarNet(nn.Module):
    def __init__(self, fbs=False, sparsity_ratio=1.0, test=False):
        super().__init__()
        self.fbs = fbs
        self.test = test
        self.layer0 = FBSConv2d(3, 64, 3, stride=1, padding=0, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)
        self.layer1 = FBSConv2d(64, 64, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)
        self.layer2 = FBSConv2d(64, 128, 3, stride=2, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)
        self.layer3 = FBSConv2d(128, 128, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)
        self.layer4 = FBSConv2d(128, 128, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)
        self.layer5 = FBSConv2d(128, 192, 3, stride=2, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)
        self.layer6 = FBSConv2d(192, 192, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)
        self.layer7 = FBSConv2d(192, 192, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio, test=test)

        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, 10)

    # TODO: get g for each layer and calculate lasso
    def forward(self, x):
        if not self.test:
            if self.fbs:
                lasso = 0
                x, g = self.layer0(x)
                lasso += g
                x, g = self.layer1(x)
                lasso += g
                x, g = self.layer2(x)
                lasso += g
                x, g = self.layer3(x)
                lasso += g
                x, g = self.layer4(x)
                lasso += g
                x, g = self.layer5(x)
                lasso += g
                x, g = self.layer6(x)
                lasso += g
                x, g = self.layer7(x)
                lasso += g
            else:
                x = self.layer0(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.layer6(x)
                x = self.layer7(x)

            x = self.pool(x)
            x = torch.flatten(x, 1)

            x = self.fc(x)
            if self.fbs:
                return x, lasso
            else:
                return x
        else:
            MACs = 0
            if self.fbs:
                lasso = 0
                x, g, macs = self.layer0(x)
                lasso += g
                MACs += macs
                x, g, macs = self.layer1(x)
                lasso += g
                MACs += macs
                x, g, macs = self.layer2(x)
                lasso += g
                MACs += macs
                x, g, macs = self.layer3(x)
                lasso += g
                MACs += macs
                x, g, macs = self.layer4(x)
                lasso += g
                MACs += macs
                x, g, macs = self.layer5(x)
                lasso += g
                MACs += macs
                x, g, macs = self.layer6(x)
                lasso += g
                MACs += macs
                x, g, macs = self.layer7(x)
                lasso += g
                MACs += macs

            else:
                x, macs = self.layer0(x)
                MACs += macs
                x, macs = self.layer1(x)
                MACs += macs
                x, macs = self.layer2(x)
                MACs += macs
                x, macs = self.layer3(x)
                MACs += macs
                x, macs = self.layer4(x)
                MACs += macs
                x, macs = self.layer5(x)
                MACs += macs
                x, macs = self.layer6(x)
                MACs += macs
                x, macs = self.layer7(x)
                MACs += macs

            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            if self.fbs:
                return x, lasso, MACs
            else:
                return x, MACs