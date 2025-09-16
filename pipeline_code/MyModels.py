import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from MyFunc import *


# ---------------- CNN 模型定义 -------------------

# TODO: BinaryCNN and CNN3L combined -> 中科大
class BinaryCNN(nn.Module):  
    '''
    参考文献：Short-term stock price trend prediction with imaging high frequency limit order book data
    '''
    def __init__(self, in_channels=1):
        super(BinaryCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),   # (1, 224, 60) -> (32, 224, 60)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                                      # -> (32, 112, 30)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # -> (64, 112, 30)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                                      # -> (64, 56, 15)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # -> (128, 56, 15)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),                                      # -> (128, 28, 7)

            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 60)
            dummy_out = self.features(dummy)
            self.fc_in = dummy_out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_in, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # 对应 CrossEntropyLoss 的2类输出
        )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN3L(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   # [64, 224, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [64, 112, 30]

            nn.Conv2d(64, 128, 3, padding=1), # [128, 112, 30]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [128, 56, 15]

            nn.Conv2d(128, 256, 3, padding=1),# [256, 56, 15]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),     # [256, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class CNN3L_Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(1, 2)),  # -> [64, 224, 60]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                          # -> [64, 112, 30]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),         # -> [128, 112, 30]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                          # -> [128, 56, 15]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),        # -> [256, 56, 15]
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))                          # -> [256, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# squeeze-and-excitation block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class CNN3L_SE(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # -> [64, 224, 60]
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2, 2),                        # -> [64, 112, 30]

            nn.Conv2d(64, 128, 3, padding=1),          # -> [128, 112, 30]
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(2, 2),                        # -> [128, 56, 15]

            nn.Conv2d(128, 256, 3, padding=1),         # -> [256, 56, 15]
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)),              # -> [256, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)                          # 对应 CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# TODO: CNN5L or CNN5L_SE
class CNN5L(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # -> [64, 224, 60]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),            # -> [64, 224, 60]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # -> [64, 112, 30]

            nn.Conv2d(64, 128, 3, padding=1),           # -> [128, 112, 30]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),          # -> [128, 112, 30]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # -> [128, 56, 15]

            nn.Conv2d(128, 256, 3, padding=1),          # -> [256, 56, 15]
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))                # -> [256, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNN5L_SE(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# TODO: VGG11
from torchvision.models import vgg11

class BinaryVGG11(nn.Module):
    def __init__(self):
        super(BinaryVGG11, self).__init__()
        # 加载 VGG11 模型的卷积部分（不包括全连接层）
        base_model = vgg11(pretrained=False)
        
        # 修改输入通道数为 1（原始是 3 通道）
        base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.features = base_model.features
        
        # 自动推导经过卷积后输出的维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 60)  # (batch, channel=1, H, W)
            features_output = self.features(dummy_input)
            flatten_dim = features_output.shape[1] * features_output.shape[2] * features_output.shape[3]
            print(f"Flatten dimension: {flatten_dim}")  # 可选：打印出来查看

        # 构建分类器部分（全连接层）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2)  # 二分类
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# TODO：ResNet+CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNetCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # [64, 112, 30]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                              # [64, 56, 15]
        )
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x



# TODO: CNN1L, CNN2L, CNN4L
class CNN1L(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   # -> [64, 224, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # -> [64, 112, 30]
            nn.AdaptiveAvgPool2d((1, 1))      # -> [64, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNN2L(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   # -> [64, 224, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # -> [64, 112, 30]

            nn.Conv2d(64, 128, 3, padding=1), # -> [128, 112, 30]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))      # -> [128, 1, 1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNN1L_Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), padding=(2, 1)),  # 更大感受野
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 只压缩高度
            nn.AdaptiveAvgPool2d((2, 2))       # 不要太早压缩成(1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),       # [64, 2, 2] = 256
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class CNN2L_Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 仍压缩但保留空间结构
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [128, 4, 4] = 2048
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN4L(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# TODO: CNN + CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        ca = self.sigmoid(ca).view(b, c, 1, 1)
        return x * ca

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.sigmoid(self.conv(sa))
        return x * sa

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class CNN3L_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            CBAM(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            CBAM(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            CBAM(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# TODO: CNN + Transformer Encoder
class CNN3L_Transformer(nn.Module):
    def __init__(self, nhead=4, num_layers=1, dim_feedforward=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   # [B, 64, 224, 60]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [B, 64, 112, 30]

            nn.Conv2d(64, 128, 3, padding=1), # [B, 128, 112, 30]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # [B, 128, 56, 15]

            nn.Conv2d(128, 256, 3, padding=1),# [B, 256, 56, 15]
            nn.ReLU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,  
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # CNN features
        x = self.cnn(x)             # [B, 256, 56, 15]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)     # [B, 256, 840]
        x = x.permute(0, 2, 1)      # [B, 840, 256] → sequence length 840

        # Transformer
        x = self.transformer(x)     # [B, 840, 256]
        x = x.mean(dim=1)           # Global Average Pooling over tokens → [B, 256]

        return self.classifier(x)


# TODO: CNN + ViT Block
class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x2 = self.attn(x, x, x)[0]
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        return self.norm2(x + x2)

class CNN_ViT(nn.Module):
    def __init__(self, patch_size=(16, 10), dim=128, depth=1, heads=4, mlp_dim=256):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        self.conv = nn.Conv2d(1, dim, kernel_size=3, padding=1)

        self.num_patches = (224 // patch_size[0]) * (60 // patch_size[1])
        self.proj = nn.Linear(dim * patch_size[0] * patch_size[1], dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        self.transformer = nn.Sequential(*[
            ViTBlock(dim, heads, mlp_dim) for _ in range(depth)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)  # -> [B, dim, 224, 60]
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_h, self.patch_h).unfold(3, self.patch_w, self.patch_w)
        x = x.contiguous().view(B, C, -1, self.patch_h, self.patch_w)
        x = x.flatten(3).permute(0, 2, 1, 3).flatten(2)  # [B, N, C*patch_h*patch_w]
        x = self.proj(x) + self.pos_embedding           # [B, N, dim]
        x = self.transformer(x)                         # [B, N, dim]
        return self.classifier(x.mean(dim=1))           # [B, dim] -> [B, 2]


# TODO: ViT for SmallImage
class ViT_SmallImage(nn.Module):
    def __init__(self, img_size=(224, 60), patch_size=(16, 12), dim=192, depth=2, heads=4, mlp_dim=384, n_classes=2):
        super().__init__()
        H, W = img_size
        ph, pw = patch_size
        self.n_patches = (H // ph) * (W // pw)
        self.patch_dim = ph * pw  # Flattened dim per patch

        self.patch_embedding = nn.Linear(self.patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = 16, 12
        # reshape into patches
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # [B, C, n_h, n_w, ph, pw]
        x = x.contiguous().view(B, C, -1, ph * pw)  # [B, C, N, P]
        x = x.squeeze(1)                            # [B, N, P] → (C=1)

        x = self.patch_embedding(x) + self.pos_embedding  # [B, N, D]
        x = self.transformer(x)                            # [B, N, D]
        x = x.mean(dim=1)                                  # [B, D]
        return self.classifier(x)


# ------------- EarlyStopping 工具类 -------------
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0