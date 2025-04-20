import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


REC_CNT=20

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
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                continue
            df = pd.read_feather(path)
            if downsample_args:
                df = sample_df(df, downsample_args)
            y = df['label'].values
            X = df.drop(columns=['label']).values
            print(f"Loaded data for {path}: {X.shape}")

            # 更新统计量
            if isinstance(mean, int):  # 第一次
                mean = np.zeros(X.shape[1])
                var = np.zeros(X.shape[1])
            mean, var, count = update_mean_std(mean, var, count, X)

            X_train_list.append(X)
            y_train_list.append(y)

    std = np.sqrt(var)
    std[std == 0] = 1e-6  # 防止除零

    # === 2. 标准化训练集 ===
    X_train = np.vstack([(X - mean) / std for X in X_train_list])
    y_train = np.concatenate(y_train_list)
    train_df = np.concatenate([X_train, y_train[:, None]], axis=1)
    print(f"train_df shape: {train_df.shape}")

    # === 3. 测试集处理（复用 mean/std） ===
    X_test_list, y_test_list = [], []
    for folder in test_data_folder_list:
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                continue
            df = pd.read_feather(path)
            if downsample_args:
                df = sample_df(df, downsample_args)
            y = df['label'].values
            X = df.drop(columns=['label']).values
            X = (X - mean) / std
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
def preprocess_data_cnn(df, is_train=True):
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # 标准化到均值0，方差1
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # std[std == 0] = 1e-6
    # X = (X - mean) / std

    # Reshape 为 CNN 输入格式
    X = X.reshape((-1, 1, 224, 3 * REC_CNT))
    
    if is_train:
        # TODO: 是不是应该时间序列交叉验证
        # X_train, X_val, y_train, y_val
        # return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        # 按时间顺序划分训练/验证集（假设 df 已按时间顺序排序）
        val_ratio = 0.3
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        return X_train, X_val, y_train, y_val
    else:
        return X, y

# ---------------- CNN 模型定义 -------------------
class CNNClassifier(nn.Module):
    def __init__(self, in_channels=1):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)  # (1, 224, 60) → (16, 224, 60)
        self.pool = nn.MaxPool2d(2, 2)                                     # → (16, 112, 30)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)           # → (32, 112, 30)
        self.fc1 = nn.Linear(32 * 56 * 15, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → (16, 112, 30)
        x = self.pool(F.relu(self.conv2(x)))  # → (32, 56, 15)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
def train_cnn(model_class, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, lr=1e-4, save_path="cnn_model.pt"):
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
    early_stopper = EarlyStopping(patience=6, delta=0.001)

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
    plt.savefig("training_plot.png")
    print("训练曲线已保存为: training_plot.png")

    return model
    
# ----------- 使用训练好的模型预测并可视化对比 -------------
def predict_and_compare(model_class, model_path, X_test, y_test, n_show=100):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=128)

    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            all_preds.extend(preds)

    print("预测结果对比 (classification_report):")
    print(classification_report(y_test, all_preds, digits=4))
    # save report
    report = classification_report(y_test, all_preds, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{model_class}classification_report.csv", index=True)

    # 可视化前 n_show 条预测
    plt.figure(figsize=(12, 4))
    plt.plot(range(n_show), y_test[:n_show], label='True Label', marker='o')
    plt.plot(range(n_show), all_preds[:n_show], label='Predicted', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.title(f'True vs Predicted Labels (前 {n_show} 个)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("prediction_comparison.png")
    print("预测对比图已保存为: prediction_comparison.png")


# ---------------- 主程序 -------------------
if __name__ == "__main__":

    code_list = [
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
    
    # train_data_folder = 'local_data/2D_data_11-1_11-7'
    # train_data_folder_2 = 'local_data/2D_data_11-8_11-14'
    # train_data_folder_3 = 'local_data/2D_data_11-15_11-21'
    # test_data_folder = 'local_data/2D_data_11-22_11-30' 

    # train_df = load_data(train_data_folder, code_list)
    # train_df_2 = load_data(train_data_folder_2, code_list)
    # train_df_3 = load_data(train_data_folder_3, code_list)
    # train_df = pd.concat([train_df, train_df_2, train_df_3], ignore_index=True)
    # X_train, X_val, y_train, y_val = preprocess_data_cnn(train_df)

    # test_df = load_data(test_data_folder, code_list, is_train=False)
    # X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
    
    downsample_args = {
        "strategy": "fraction",
        "value": 0.5,
        "random_state": 42
    }

    train_df, test_df = load_data(
        train_data_folder_list=[
            "local_data/2D_data_11-1_11-7",
            "local_data/2D_data_11-8_11-14",
            "local_data/2D_data_11-15_11-21"
        ],
        test_data_folder_list=["local_data/2D_data_11-22_11-30"],
        code_list=code_list,
        downsample_args=downsample_args
    )

    X_train, X_val, y_train, y_val  = preprocess_data_cnn(train_df)
    X_test, y_test = preprocess_data_cnn(test_df, is_train=False)

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")

    print("开始训练 CNN...")
    model_class = VGG7
    epochs = 50
    batch_size = 128
    lr = 1e-3
    save_path = "cnn_model.pt"
    print(f"模型: {model_class.__name__}")
    print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    # cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
    #                       epochs, batch_size, lr, save_path)
    
    print("评估 CNN...")
    predict_and_compare(model_class, save_path, X_test, y_test)
