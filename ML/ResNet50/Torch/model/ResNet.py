import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        else:
            self.shortcut = []
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + self.shortcut(x)
        return F.relu(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_blocks = [3,4,6,3]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_block(64, num_blocks[0], 1)
        self.layer2 = self._make_block(128, num_blocks[1], 2)
        self.layer3 = self._make_block(256, num_blocks[2], 2)
        self.layer4 = self._make_block(512, num_blocks[3], 2)

        self.linear = nn.Linear(512*ResNetBottleneck.expansion, num_classes)
        
    
    def _make_block(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBottleneck(self.in_channels, planes, stride))
            self.in_channels = planes * ResNetBottleneck.expansion
            return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
