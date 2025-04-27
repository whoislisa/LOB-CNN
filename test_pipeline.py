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


   
# ----------- 使用训练好的模型预测并可视化对比 -------------

def evaluate_and_save(y_true, y_pred, model_path, tag=""):

    print(f"评估 {tag} ...")
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{model_path[:-3]}_{tag}_classification_report.csv", index=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {tag}")
    plt.savefig(f"{model_path[:-3]}_{tag}_confusion_matrix.png")
    plt.close()


def predict_and_collect(model_class, model_path, X_test, y_test, week_idx):
    
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=128)

    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(y_test)
    print(f"第 {week_idx+1} 周平均 Cross-Entropy Loss: {avg_loss:.4f}")
    with open(f"{model_path[:-3]}_week{week_idx+1}_loss.txt", "w") as f:
        f.write(f"Average CrossEntropy Loss: {avg_loss:.4f}\n")

    # 每周单独保存指标
    evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=model_path, tag=f"week{week_idx+1}")

    return all_preds, all_labels


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
    

    # #### 1 ####
    # model_class = CNN3L
    # # 训练好的模型的存储路径
    # save_path = "model_res_for_backtest/M_CNN3L_e20_b128_l0.0004_D_CSI2000_T_0427-0313/M_CNN3L_e20_b128_l0.0004_D_CSI2000_T_0427-0313.pt"
    # print(f"数据: {data_args['name']}")
    # print(f"模型: {model_class.__name__}")
    # print(f"保存路径: {save_path}")

    # print("评估 CNN...")
    # all_preds = []
    # all_labels = []
    # for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
    #     print(f"加载第 {week_idx+1} 周的数据...")
    #     test_df = load_data(
    #         data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
    #         code_list=data_args['code_list'],
    #     )
    #     X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
    #     print(f"本周测试集大小: {X_test.shape}")
    #     print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

    #     preds, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
    #     all_preds.extend(preds)
    #     all_labels.extend(labels)

    # print("整体评估（全月数据）...")
    # evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=save_path, tag="overall")

  
    
    # #### 2 ####
    # model_class = CNN3L_SE
    # # 训练好的模型的存储路径
    # save_path = "M_CNN3L_SE_e20_b128_l0.001_D_CSI2000_T_0427-1022.pt"
    # print(f"数据: {data_args['name']}")
    # print(f"模型: {model_class.__name__}")
    # print(f"保存路径: {save_path}")

    # print("评估 CNN...")
    # all_preds = []
    # all_labels = []
    # for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
    #     print(f"加载第 {week_idx+1} 周的数据...")
    #     test_df = load_data(
    #         data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
    #         code_list=data_args['code_list'],
    #     )
    #     X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
    #     print(f"本周测试集大小: {X_test.shape}")
    #     print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

    #     preds, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
    #     all_preds.extend(preds)
    #     all_labels.extend(labels)

    # print("整体评估（全月数据）...")
    # evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=save_path, tag="overall")
    

    # # #### 3 ####
    # model_class = CNN5L
    # # 训练好的模型的存储路径
    # save_path = "model_res_for_backtest/M_CNN5L_e20_b128_l0.001_D_CSI2000_T_0427-0439/M_CNN5L_e20_b128_l0.001_D_CSI2000_T_0427-0439.pt"
    # print(f"数据: {data_args['name']}")
    # print(f"模型: {model_class.__name__}")
    # print(f"保存路径: {save_path}")
    
    # print("评估 CNN...")
    # all_preds = []
    # all_labels = []
    # for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
    #     print(f"加载第 {week_idx+1} 周的数据...")
    #     test_df = load_data(
    #         data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
    #         code_list=data_args['code_list'],
    #     )
    #     X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
    #     print(f"本周测试集大小: {X_test.shape}")
    #     print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

    #     preds, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
    #     all_preds.extend(preds)
    #     all_labels.extend(labels)

    # print("整体评估（全月数据）...")
    # evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=save_path, tag="overall") 


    
    # #### 4 ####
    # model_class = BinaryVGG11
    # # 训练好的模型的存储路径
    # save_path = "model_res_for_backtest/M_BinaryVGG11_e20_b64_l0.0004_D_CSI2000_T_0427-0535/M_BinaryVGG11_e20_b64_l0.0004_D_CSI2000_T_0427-0535.pt"
    # print(f"数据: {data_args['name']}")
    # print(f"模型: {model_class.__name__}")
    # print(f"保存路径: {save_path}")

    # print("评估 CNN...")
    # all_preds = []
    # all_labels = []
    # for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
    #     print(f"加载第 {week_idx+1} 周的数据...")
    #     test_df = load_data(
    #         data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
    #         code_list=data_args['code_list'],
    #     )
    #     X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
    #     print(f"本周测试集大小: {X_test.shape}")
    #     print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

    #     preds, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
    #     all_preds.extend(preds)
    #     all_labels.extend(labels)

    # print("整体评估（全月数据）...")
    # evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=save_path, tag="overall")
    
    
    
    #### 5 ####
    model_class = ResNetCNN
    # 训练好的模型的存储路径
    save_path = "M_ResNetCNN_e20_b128_l0.001_D_CSI2000_T_0427-1123.pt"
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_labels = []
    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=save_path, tag="overall")

    #### 5 ####
    model_class = ResNetCNN
    # 训练好的模型的存储路径
    save_path = "M_ResNetCNN_e20_b128_l0.0004_D_CSI2000_T_0427-1202.pt"
    print(f"数据: {data_args['name']}")
    print(f"模型: {model_class.__name__}")
    print(f"保存路径: {save_path}")

    print("评估 CNN...")
    all_preds = []
    all_labels = []
    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data_cnn(test_df, is_train=False)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, labels = predict_and_collect(model_class, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=save_path, tag="overall")
