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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


REC_CNT=20
IS_BINARY=False

# ---------------- 数据加载 -------------------
# def load_data(data_folder, code_list, is_train=True):
#     all_df = []
#     for code in code_list:
#         # --- feather format ---
#         file_path = os.path.join(data_folder, f"{code}.feather")
#         if os.path.exists(file_path):
#             df = pd.read_feather(file_path)
#             all_df.append(df)
#             print(f"Loaded data for {file_path}: {df.shape}")

#     all_df = pd.concat(all_df, ignore_index=True)
    
#     if is_train:  # 训练集降采样
#         all_df = resample(all_df, n_samples=all_df.shape[0]//2, random_state=42)
    
#     return all_df
def load_data(train_data_folder_list, test_data_folder_list, code_list, downsample_args=None):
    def sample_df(df, args):
        if args is None:
            return df
        if args['strategy'] == 'fraction':
            return df.sample(frac=args['value'], random_state=args.get('random_state', 42))
        elif args['strategy'] == 'fixed':
            n_samples = min(args['value'], len(df))
            return df.sample(n=n_samples, random_state=args.get('random_state', 42))
        return df

    def update_mean_std(mean, var, count, new_data):
        n = count + len(new_data)
        new_mean = (mean * count + new_data.sum(axis=0)) / n
        new_var = (var * count + ((new_data - mean) ** 2).sum(axis=0)) / n
        return new_mean, new_var, n

    # === 1. 训练集处理（含 mean/std 计算） ===
    mean, var, count = 0, 0, 0
    X_train_list, y_train_list = [], []

    for folder in train_data_folder_list:
        if not IS_BINARY:
            folder = "multiclass_" + folder
            
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            df = pd.read_feather(path)
            if downsample_args:
                df = sample_df(df, downsample_args)
            y = df['label'].values
            X = df.drop(columns=['label']).values
            print(f"Loaded data for {path}: {X.shape}")

            # # 更新统计量
            # if isinstance(mean, int):  # 第一次
            #     mean = np.zeros(X.shape[1])
            #     var = np.zeros(X.shape[1])
            # mean, var, count = update_mean_std(mean, var, count, X)

            X_train_list.append(X)
            y_train_list.append(y)

    # std = np.sqrt(var)
    # std[std == 0] = 1e-6  # 防止除零

    # # === 2. 标准化训练集 ===
    # X_train = np.vstack([(X - mean) / std for X in X_train_list])
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    train_df = np.concatenate([X_train, y_train[:, None]], axis=1)
    print(f"train_df shape: {train_df.shape}")

    # === 3. 测试集处理（复用 mean/std） ===
    X_test_list, y_test_list = [], []
    for folder in test_data_folder_list:
        if not IS_BINARY:
            folder = "multiclass_" + folder
        
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            df = pd.read_feather(path)
            if downsample_args:
                df = sample_df(df, downsample_args)
            y = df['label'].values
            X = df.drop(columns=['label']).values
            # X = (X - mean) / std
            print(f"Loaded data for {path}: {X.shape}")
            X_test_list.append(X)
            y_test_list.append(y)

    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    test_df = np.concatenate([X_test, y_test[:, None]], axis=1)
    print(f"test_df shape: {test_df.shape}")

    # 转回 DataFrame，保持兼容性
    columns = [f"f{i}" for i in range(X_train.shape[1])] + ["label"]
    train_df = pd.DataFrame(train_df, columns=columns)
    test_df = pd.DataFrame(test_df, columns=columns)
    
    return train_df, test_df



# ---------------- CNN 数据预处理 -------------------

def preprocess_data_cnn(df, is_train=True, is_DeepVol=False, balanced=False):
    X = df.drop(columns=['label']).values
    X = X.reshape((-1, 1, 224, 60))
    
    y = df['label'].values
    if not IS_BINARY:
        y[y == -1] = 2  # label -1 --> 2

    if is_train:
        if balanced:
            # --------- 下采样 majority class 开始 ---------
            from collections import Counter
            counter = Counter(y)
            if len(counter) == 2:  # 二分类
                # 找到 minority class 和 majority class
                minority_class = min(counter, key=counter.get)
                majority_class = max(counter, key=counter.get)

                # 获取原始索引
                idx_min = np.where(y == minority_class)[0]
                idx_maj = np.where(y == majority_class)[0]

                # 按时间顺序从多数类中抽样，不打乱
                idx_maj_down = np.sort(np.random.choice(idx_maj, size=len(idx_min), replace=False))

                # 合并新索引后按时间顺序排列
                idx_combined = np.sort(np.concatenate([idx_min, idx_maj_down]))

                # 保留原始时间顺序
                X = X[idx_combined]
                y = y[idx_combined]
            # --------- 下采样 majority class 结束 ---------
        
        val_ratio = 0.3
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        return X_train, X_val, y_train, y_val
    
    else:
        return X, y


# ---------------- CNN 模型定义 -------------------

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


class BinaryCNN_2(nn.Module):
    '''
    参考文献：Short-term stock price trend prediction with imaging high frequency limit order book data
    '''
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: [1, 224, 60]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # -> [64, 224, 60]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),                         # -> [64, 112, 30]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> [128, 112, 30]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),                         # -> [128, 56, 15]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# -> [256, 56, 15]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),                 # -> [256, 1, 1]
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # -> [256]
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)                              # 多分类可选；二分类时改为1+sigmoid
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class VGG7(nn.Module):
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

class BinaryCNN_3(nn.Module):
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

class MulticlassCNN_3(nn.Module):
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
            nn.Linear(128, 3)                          # 对应 CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeepVOL(nn.Module):
    '''
    参考文献：The short-term predictability of returns in order book markets- A deep learning perspective
    '''
    def __init__(self, num_classes=2):
        super(DeepVOL, self).__init__()
        # 输入为 (B, 1, 20, 224, 2)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(1, 3, 2), padding=(0, 1, 0)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 1))  # 减半224 → 112
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1))  # 减半时间20 → 10，宽112 → 56
        )

        self.flatten = nn.Flatten()
        # self.fc1 = nn.Sequential(
        #     nn.Linear(32 * 10 * 56 * 2, 128),  # 乘法维度对应输出 shape
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )
        
        # 自动推理 fc 输入维度
        self._fc_input_dim = None
        self.fc1 = nn.Identity()  # 先定义空的 fc1，等 forward 时动态定义

        self.output = nn.Linear(128, num_classes)

    def forward(self, x):  # x: (B, 1, 20, 224, 2)
        x = self.conv1(x)  # -> (B, 16, 20, 112, 2)
        x = self.conv2(x)  # -> (B, 32, 10, 56, 2)
        x = self.flatten(x)
        
        # 第一次 forward 时定义 fc1
        if isinstance(self.fc1, nn.Identity):
            self._fc_input_dim = x.shape[1]
            self.fc1 = nn.Sequential(
                nn.Linear(self._fc_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3)
            ).to(x.device)
            self.add_module("fc1", self.fc1)

        
        x = self.fc1(x)
        x = self.output(x)
        return x


from torchvision.models import vgg16

class BinaryVGG16(nn.Module):
    def __init__(self):
        super(BinaryVGG16, self).__init__()
        
        # 载入VGG16模型结构，不加载预训练权重（可选：weights=VGG16_Weights.IMAGENET1K_V1）
        self.vgg_features = vgg16(weights=None).features
        
        # 自定义第一个卷积层：替换原始3通道为1通道
        self.vgg_features[0] = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=3, stride=1, padding=1
        )

        # 自定义全连接层，二分类（原始VGG16是1000类）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 1, 4096),  # 假设输入为 (1, 224, 60)，经过池化后大概是 (512, 7, 1)
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)  # 二分类
        )

    def forward(self, x):
        x = self.vgg_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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

# ---------------- CNN 训练 + 保存 + 绘图 -------------------
def train_cnn(model_class, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, lr=1e-4, early_stop_cnt = 6, save_path="cnn_model.pt"):
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)),
                            batch_size=batch_size)

    train_losses = []
    val_f1s = []
    early_stopper = EarlyStopping(patience=early_stop_cnt, delta=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # === 验证集 F1 ===
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb)
                pred = torch.argmax(pred, dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(yb.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f} | Val F1={val_f1:.4f}")

        # === EarlyStopping 检查 ===
        early_stopper(val_f1)
        if early_stopper.early_stop:
            print("触发 EarlyStopping，停止训练")
            break

    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

    # === 绘图 ===
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_f1s, label='Val Weighted F1')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Training Loss & Validation F1')
    plt.grid()
    plt.tight_layout()
    
    train_plot_path = save_path[:-3] + "_training_plot.png"
    plt.savefig(train_plot_path)
    print("训练曲线已保存为: ", train_plot_path)

    return model
    
# ----------- 使用训练好的模型预测并可视化对比 -------------
# def predict_and_compare(model_class, model_path, X_test, y_test, n_show=100):
#     model = model_class().to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
#                                            torch.tensor(y_test, dtype=torch.long)),
#                              batch_size=128)

#     all_preds = []
#     with torch.no_grad():
#         for xb, _ in test_loader:
#             xb = xb.to(device)
#             preds = model(xb)
#             preds = torch.argmax(preds, dim=1).cpu().numpy()
#             all_preds.extend(preds)

#     print("预测结果对比 (classification_report):")
#     print(classification_report(y_test, all_preds, digits=4))
    
#     # save report
#     report = classification_report(y_test, all_preds, digits=4, output_dict=True)
#     report_df = pd.DataFrame(report).transpose()
#     report_df.to_csv(f"{model_path[:-3]}_classification_report.csv", index=True)

#     # 可视化前 n_show 条预测
#     plt.figure(figsize=(12, 4))
#     plt.plot(range(n_show), y_test[:n_show], label='True Label', marker='o')
#     plt.plot(range(n_show), all_preds[:n_show], label='Predicted', marker='x')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Class')
#     plt.title(f'True vs Predicted Labels (前 {n_show} 个)')
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
    
#     plt.savefig(f"{model_path[:-3]}_prediction_comparison.png")
# 新增 loss 和混淆矩阵
def predict_and_compare(model_class, model_path, X_test, y_test, n_show=100):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=128)

    all_preds = []
    total_loss = 0 
    criterion = nn.CrossEntropyLoss(reduction='sum')
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(yb.cpu().numpy())

    # 计算平均 loss
    avg_loss = total_loss / len(y_test)
    print(f"Test Set Average Cross-Entropy Loss: {avg_loss:.4f}")
    with open(f"{model_path[:-3]}_loss.txt", "w") as f:
        f.write(f"Average CrossEntropy Loss: {avg_loss:.4f}\n")

    # 分类报告
    print("预测结果对比:")
    print(classification_report(y_test, all_preds, digits=4))
    report = classification_report(y_test, all_preds, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{model_path[:-3]}_classification_report.csv", index=True)

    # 混淆矩阵图
    cm = confusion_matrix(y_test, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {model_path[:-3]}")
    plt.savefig(f"{model_path[:-3]}_confusion_matrix.png")
    plt.close()


# ---------------- 主程序 -------------------
if __name__ == "__main__":

    code_list_CSI50 = [
        # 中证A50十大权重
        '600519sh',  # 贵州茅台
        '300750sz',  # 宁德时代
        '601318sh',  # 中国平安
        '600036sh',  # 招商银行
        '600900sh',  # 长江电力
        '000333sz',  # 美的集团
        '002594sz',  # 比亚迪
        '601899sh',  # 紫金矿业
        '600030sh',  # 中信证券
        '600276sh',  # 恒瑞医药
    ]
    
    code_list_CSI2000 = [
        '002031sz',  # 巨轮智能
        '300766sz',  # 每日互动
        '300377sz',  # 赢时胜
        '300353sz',  # 东土科技
        '300100sz',  # 双林股份
        '300184sz',  # 力源信息
        '300276sz',  # 三丰智能
        '603009sh',  # 北特科技
        '002379sz',  # 宏创控股
        '300718sz',  # 长盛轴承
    ]
    
    data_args_CSI50 = {
        'name': 'CSI50',
        'code_list': code_list_CSI50,
        'train_data_folder_list': [
            "data_202111/2D_data_11-1_11-7",
            "data_202111/2D_data_11-8_11-14",
            "data_202111/2D_data_11-15_11-21"
        ],
        'test_data_folder_list': ["data_202111/2D_data_11-22_11-30"],
    }
    
    data_args_CSI2000 = {
        'name': 'CSI2000',
        'code_list': code_list_CSI2000,
        'train_data_folder_list': [
            "data2_202111/2D_data_11-1_11-7",
            "data2_202111/2D_data_11-8_11-14",
            "data2_202111/2D_data_11-15_11-21"
        ],
        'test_data_folder_list': ["data2_202111/2D_data_11-22_11-30"],
    }
    
    
    data_args = data_args_CSI50
    
    downsample_args = {
        "strategy": "fraction",
        "value": 0.5,
        "random_state": 42
    }

    train_df, test_df = load_data(
        train_data_folder_list=data_args['train_data_folder_list'],
        test_data_folder_list=data_args['test_data_folder_list'],
        code_list=data_args['code_list'],
        downsample_args=downsample_args
    )

    X_train, X_val, y_train, y_val  = preprocess_data_cnn(train_df, balanced=False)    
    X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"验证集标签分布: {np.unique(y_val, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")

    ### Round 1
    print("开始训练 CNN...")
    # model_class = BinaryCNN_3  # faster and consistent
    # model_class = BinaryVGG11
    model_class = MulticlassCNN_3
    epochs = 20
    batch_size = 128
    lr = 1e-3
    early_stop_cnt = 4
    
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    cur_time = datetime.now().strftime("%m%d-%H%M")
    name_str = f"M_{model_class.__name__}_e{epochs}_b{batch_size}_l{lr}_D_{data_args['name']}_T_{cur_time}"
    save_path = name_str + ".pt"
    cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
                          epochs, batch_size, lr, early_stop_cnt, save_path)
    
    print("评估 CNN...")
    predict_and_compare(model_class, save_path, X_test, y_test)

