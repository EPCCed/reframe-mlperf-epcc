import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm2d(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass
    def forward(self, x):
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.in_planes = 64
        num_blocks = [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(Bottleneck, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], 2)
        self.linear = nn.Linear(512*Bottleneck.expansion,num_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, (3, 3), (2, 2), (1, 1))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet50()
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, data):
        x, y = data
        logits = self.model(x)
        return self.criterion(logits, y)
