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


REC_CNT=20
IS_BINARY=True  # Unfortunately, multiclass does not work well with the current model.


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

            if not is_Nov:  # December
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
def preprocess_data_cnn(df, is_train=True, balanced=False):
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


# ---------------- 测试数据加载、处理 -------------------
def load_test_data(data_folder_list, code_list):

    df_list = []
    columns = []
    for folder in data_folder_list:           
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            df = pd.read_feather(path)
            df_list.append(df)
            columns = df.columns.tolist()
            print(f"Loaded data for {path}: {df.shape}")

    full_df = pd.concat(df_list, axis=0)
    full_df.columns = columns

    print(f"df shape: {full_df.shape}")
    return full_df

def preprocess_test_data_cnn(df):
    X = df.drop(columns=['label_2', 'label_3', 'datetime', 'Code', 'mid_price']).values
    X = X.reshape((-1, 1, 224, 60))
    
    y = df['label_2'].values
    info_df = df[['datetime', 'Code', 'mid_price']].copy()

    return X, y, info_df
