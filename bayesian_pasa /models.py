import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import PASA, BayesianPASA, Mish, Swish
from .normalization import StandardLayerNorm, RLayerNorm, BayesianRLayerNorm

class EfficientCNN(nn.Module):
    """
    CNN with configurable normalization and activation for CIFAR-10/100.
    """
    def __init__(self, norm_type='layer', activation='relu', num_classes=10):
        super().__init__()
        self.norm_type = norm_type
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.norm1 = self._create_norm_layer(16)
        self.act1 = self._create_activation()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.norm2 = self._create_norm_layer(32)
        self.act2 = self._create_activation()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm3 = self._create_norm_layer(64)
        self.act3 = self._create_activation()
        self.pool3 = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self.return_weights = False

    def _create_norm_layer(self, num_features):
        if self.norm_type == 'layer':
            return StandardLayerNorm(num_features)
        elif self.norm_type == 'r_layer':
            return RLayerNorm(num_features)
        elif self.norm_type == 'bayesian_r_layer':
            return BayesianRLayerNorm(num_features)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")

    def _create_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leakyrelu':
            return nn.LeakyReLU(0.01)
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'swish':
            return Swish()
        elif self.activation == 'mish':
            return Mish()
        elif self.activation == 'pasa':
            return PASA()
        elif self.activation == 'bayesian_pasa':
            return BayesianPASA()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x, return_weights=False):
        self.return_weights = return_weights

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        if self.return_weights and isinstance(self.act3, (PASA, BayesianPASA)):
            x, weights = self.act3(x, return_weights=True)
        else:
            x = self.act3(x)
            weights = None

        x = self.pool3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if weights is not None:
            return x, weights
        return x


class BasicBlock(nn.Module):
    """Basic Block for ResNet-18"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, activation='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = activation

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def _get_activation(self):
        if self.activation == 'relu':
            return F.relu
        elif self.activation == 'gelu':
            return F.gelu
        elif self.activation == 'swish':
            return lambda x: x * torch.sigmoid(x)
        elif self.activation == 'mish':
            return lambda x: x * torch.tanh(F.softplus(x))
        elif self.activation == 'pasa':
            return lambda x: PASA()(x)  # Note: This creates new instance each time - better to pre-initialize
        else:
            return F.relu

    def forward(self, x):
        act = self._get_activation()
        out = act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = act(out)
        return out


class ResNet18_CIFAR100(nn.Module):
    """
    ResNet-18 adapted for CIFAR-100 (32x32 images, 100 classes)
    """
    def __init__(self, activation='relu', num_classes=100):
        super(ResNet18_CIFAR100, self).__init__()
        self.in_planes = 64
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, activation=self.activation))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
