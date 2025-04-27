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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

 
REC_CNT=20
IS_BINARY=True  
# Unfortunately, multiclass does not work well with the current model.

# ---------------- 数据加载 -------------------
def load_data(data_folder_list, code_list, downsample_args=None, is_Nov=False):
    def sample_df(df, args):
        if args is None:
            return df
        if args['strategy'] == 'fraction':
            return df.sample(frac=args['value'], random_state=args.get('random_state', 42))
        elif args['strategy'] == 'fixed':
            n_samples = min(args['value'], len(df))
            return df.sample(n=n_samples, random_state=args.get('random_state', 42))
        return df

    X_list, y_list = [], []
    if not IS_BINARY and is_Nov:
        data_folder_list = ["multiclass_"+path for path in data_folder_list]
    for folder in data_folder_list:           
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            # print(f"Loading data from {path}...")
            df = pd.read_feather(path)
            # print(f"Loaded data for {path}: {df.shape}")
            # print(df.columns)
            # exit()
            if downsample_args:
                df = sample_df(df, downsample_args)

            if not is_Nov:
                if IS_BINARY:
                    y = df['label_2'].values
                else:
                    y = df['label_3'].values
                X = df.drop(columns=['label_2', 'label_3']).values
            else:
                y = df['label'].values
                X = df.drop(columns=['label']).values

            print(f"Loaded data for {path}: {X.shape}")

            X_list.append(X)
            y_list.append(y)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    df = np.concatenate([X, y[:, None]], axis=1)
    print(f"df shape: {df.shape}")

    # 转回 DataFrame，保持兼容性
    columns = [f"f{i}" for i in range(X.shape[1])] + ["label"]
    df = pd.DataFrame(df, columns=columns)
    
    return df


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
def predict_and_compare(model_class, model_path, X_test, y_test):
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



# ---------------- 测试+模拟交易回测 -------------------
def simulate_backtest(
    probs,               # (N,) array，模型预测的上涨概率
    true_labels,         # (N,) array，真实涨跌标签，1表示上涨，0表示下跌
    returns,             # (N,) array，每步的真实收益率
    timestamps,          # (N,) array，每步对应的时间戳，要求是pandas.Timestamp或字符串
    max_volume=1,        # 最大挂单量，单位：手
    fill_probability=0.9 # 挂单成交概率，0-1之间
):
    """
    模拟交易回测，基于模型上涨概率进行动态仓位调整。
    """
    # 初始账户状态
    cash = 1e6   # 初始现金设置为100万
    position = 0.0
    equity_curve = []
    positions_record = []
    fill_attempts = 0
    fills_successful = 0
    trade_records = []   # 成交记录：(时间戳, 成交方向, 成交量, 成交时刻收益率)

    timestamps = pd.to_datetime(timestamps)  # 确保时间戳统一
    daily_pnl = {}
    current_day = timestamps[0].date()
    daily_return = 0.0

    for prob, label, ret, ts in zip(probs, true_labels, returns, timestamps):
        signal_volume = 0.0
        direction = 0
        
        if prob > 0.5:
            direction = 1  # 买入信号
            signal_volume = (prob - 0.5) * 2 * max_volume
        elif prob < 0.5:
            direction = -1 # 卖出信号
            signal_volume = (0.5 - prob) * 2 * max_volume
        else:
            direction = 0
            signal_volume = 0

        # 尝试挂单
        if signal_volume > 0:
            fill_attempts += 1
            if np.random.rand() < fill_probability:
                fills_successful += 1
                trade_volume = direction * signal_volume
                # 成交，更新持仓
                position += trade_volume
                trade_records.append((ts, direction, signal_volume, ret))  # 记录真实成交及收益率

        # 按当前持仓实时盈亏
        pnl = position * ret
        cash += pnl

        # 每日统计净值变化
        if ts.date() != current_day:
            daily_pnl[current_day] = daily_return
            daily_return = 0.0
            current_day = ts.date()
        daily_return += pnl

        equity_curve.append(cash)
        positions_record.append(position)

    # 收尾
    if current_day not in daily_pnl:
        daily_pnl[current_day] = daily_return

    equity_curve = np.array(equity_curve)
    positions_record = np.array(positions_record)

    ## ======= 计算回测指标 ======= ##
    daily_returns = pd.Series(daily_pnl).sort_index()
    daily_returns = daily_returns / (1e6)  # 初始本金标准化，始终是100万

    # 年化收益率
    arr = (np.prod(1 + daily_returns))**(252/len(daily_returns)) - 1

    # 夏普比率
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-6)) * np.sqrt(252)

    # 平均持仓时间
    if len(trade_records) >= 2:
        holding_durations = []
        current_pos = 0
        last_time = None
        for t, direction, volume, _ in trade_records:
            if current_pos == 0:
                last_time = t
            current_pos += direction * volume
            if np.isclose(current_pos, 0) and last_time is not None:
                duration = (t - last_time).total_seconds() / 60  # 单位分钟
                holding_durations.append(duration)
                last_time = None
        avg_holding_period = np.mean(holding_durations) if holding_durations else 0
    else:
        avg_holding_period = 0

    # 成交率
    fill_rate = fills_successful / (fill_attempts + 1e-6)

    # 胜率（只基于实际成交）
    wins = 0
    for ts, direction, volume, ret in trade_records:
        if direction * ret > 0:
            wins += 1
    win_ratio = wins / (len(trade_records) + 1e-6)

    # 最大回撤
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-6)
    mdd = drawdown.min()

    return {
        'Annualized Return': arr,
        'Sharpe Ratio': sharpe,
        'Average Holding Period (min)': avg_holding_period,
        'Fill Rate': fill_rate,
        'Winning Ratio': win_ratio,
        'Maximum Drawdown': mdd,
        'Equity Curve': equity_curve,  # 可选返回，用来画净值曲线
    }

def predict_and_backtest(model_class, model_path, X_test, y_test, returns=None, timestamps=None, mode="classification"):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=128)

    all_preds = []
    all_probs = []
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
            
            probs = torch.softmax(preds, dim=1)[:, 1]  # 概率是softmax后类别1的概率
            all_probs.extend(probs.cpu().numpy()) 

    # 计算平均 loss
    avg_loss = total_loss / len(y_test)
    print(f"Test Set Average Cross-Entropy Loss: {avg_loss:.4f}")
    with open(f"{model_path[:-3]}_loss.txt", "w") as f:
        f.write(f"Average CrossEntropy Loss: {avg_loss:.4f}\n")

    # ==== 分类指标 ====
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

    if mode == "classification":
        return all_preds, all_labels

    # ==== 回测指标（如果是 backtest 模式）====
    if mode == "backtest":
        if returns is None or timestamps is None:
            raise ValueError("returns和timestamps在backtest模式下必须提供。")
        print("开始计算回测指标...")

        probs = np.array(all_probs)
        rets = simulate_backtest(
            probs=probs,
            true_labels=np.array(y_test),
            returns=np.array(returns),
            timestamps=np.array(timestamps)
        )
        print(f"回测指标：年化收益率={rets['Annualized Return']:.4f}，夏普比率={rets['Sharpe Ratio']:.4f}")

        backtest_result_path = model_path[:-3] + "_backtest_metrics.txt"
        with open(backtest_result_path, "w") as f:
            for k, v in rets.items():
                if isinstance(v, (float, int, np.floating)):
                    f.write(f"{k}: {v:.4f}\n")

        return all_preds, all_labels

def evaluate_monthly(save_path, week_dates):
    """根据周预测结果，计算整月指标"""
    all_preds = []
    all_labels = []

    for week_name in week_dates:
        preds = np.load(f"{save_path[:-3]}_preds_{week_name}.npy")
        labels = np.load(f"{save_path[:-3]}_labels_{week_name}.npy")
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 重新计算分类指标
    print("整月分类指标:")
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # 画整月混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Monthly Confusion Matrix")
    plt.savefig(f"{save_path[:-3]}_monthly_confusion_matrix.png")
    plt.close()

    return report_df



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
        'val_data_folder_list': [
            "data_202111/2D_data_11-22_11-30"
        ],
        'test_data_folder_list': [
            "data_202112/2D_data_12-1_12-12",
            "data_202112/2D_data_12-13_12-19",
            "data_202112/2D_data_12-20_12-26",
            "data_202112/2D_data_12-27_12-31"
        ]
    }
    
    data_args_CSI2000 = {
        'name': 'CSI2000',
        'code_list': code_list_CSI2000,
        'train_data_folder_list': [
            "data2_202111/2D_data_11-1_11-7",
            "data2_202111/2D_data_11-8_11-14",
            "data2_202111/2D_data_11-15_11-21"
        ],
        'val_data_folder_list': [
            "data2_202111/2D_data_11-22_11-30"
        ],
        'test_data_folder_list': [
            "data2_202112/2D_data_12-1_12-12",
            "data2_202112/2D_data_12-13_12-19",
            "data2_202112/2D_data_12-20_12-26",
            "data2_202112/2D_data_12-27_12-31"
        ]
    }
    
    
    data_args = data_args_CSI2000

    downsample_args = {
        "strategy": "fraction",
        "value": 0.5,
        "random_state": 42
    }
    
    
    train_df = load_data(
        data_folder_list=data_args['train_data_folder_list'],
        code_list=data_args['code_list'],
        downsample_args=downsample_args,
        is_Nov=True
    )
    val_df = load_data(
        data_folder_list=data_args['val_data_folder_list'],
        code_list=data_args['code_list'],
        downsample_args=downsample_args,
        is_Nov=True
    )

    X_train, X_val, y_train, y_val  = preprocess_data_cnn(train_df, balanced=False)    
    
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"验证集标签分布: {np.unique(y_val, return_counts=True)}")
    
    print("开始训练 CNN...")
    
    
    ### 1 ####
    model_class = CNN3L_SE
    epochs = 20
    batch_size = 128
    lr = 1e-3
    early_stop_cnt = 4
    
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    cur_time = datetime.now().strftime("%m%d-%H%M")
    name_str = f"M_{model_class.__name__}_e{epochs}_b{batch_size}_l{lr}_D_{data_args['name']}_T_{cur_time}"
    print(f"保存路径: {name_str}")
    save_path = name_str + ".pt"
    cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
                          epochs, batch_size, lr, early_stop_cnt, save_path)
    
    #### 2 ####
    model_class = CNN3L_SE
    epochs = 20
    batch_size = 128
    lr = 1e-3
    early_stop_cnt = 4
    
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    cur_time = datetime.now().strftime("%m%d-%H%M")
    name_str = f"M_{model_class.__name__}_e{epochs}_b{batch_size}_l{lr}_D_{data_args['name']}_T_{cur_time}"
    print(f"保存路径: {name_str}")
    save_path = name_str + ".pt"
    cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
                          epochs, batch_size, lr, early_stop_cnt, save_path)
    
    # #### 3 ####
    # model_class = ResNetCNN
    # epochs = 20
    # batch_size = 128
    # lr = 1e-3
    # early_stop_cnt = 4
    
    # print(f"数据: {data_args['name']}")
    # print(f"模型: {model_class.__name__}")
    # print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    # cur_time = datetime.now().strftime("%m%d-%H%M")
    # name_str = f"M_{model_class.__name__}_e{epochs}_b{batch_size}_l{lr}_D_{data_args['name']}_T_{cur_time}"
    # print(f"保存路径: {name_str}")
    # save_path = name_str + ".pt"
    # cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
    #                       epochs, batch_size, lr, early_stop_cnt, save_path)
    
    # #### 4 ####
    # model_class = ResNetCNN
    # epochs = 20
    # batch_size = 128
    # lr = 4e-4
    # early_stop_cnt = 4
    
    # print(f"数据: {data_args['name']}")
    # print(f"模型: {model_class.__name__}")
    # print(f"超参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    # cur_time = datetime.now().strftime("%m%d-%H%M")
    # name_str = f"M_{model_class.__name__}_e{epochs}_b{batch_size}_l{lr}_D_{data_args['name']}_T_{cur_time}"
    # print(f"保存路径: {name_str}")
    # save_path = name_str + ".pt"
    # cnn_model = train_cnn(model_class, X_train, y_train, X_val, y_val, 
    #                       epochs, batch_size, lr, early_stop_cnt, save_path)
    
