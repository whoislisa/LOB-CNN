import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from datetime import datetime

import utils as _U

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample



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
#         all_df = resample(all_df, n_samples=50000, random_state=42)
    
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

    # === 3. 测试集处理（复用 mean/std） ===
    X_test_list, y_test_list = [], []
    for folder in test_data_folder_list:
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                continue
            df = pd.read_feather(path)
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

    # 转回 DataFrame，保持兼容性
    columns = [f"f{i}" for i in range(X_train.shape[1])] + ["label"]
    train_df = pd.DataFrame(train_df, columns=columns)
    print(f"train_df shape: {train_df.shape}")
    test_df = pd.DataFrame(test_df, columns=columns)
    print(f"test_df shape: {test_df.shape}")
    
    return train_df, test_df


def preprocess_data_baseline(df, is_train=True):
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # # 标准化到均值0，方差1
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # std[std == 0] = 1e-6  # 防止除以0
    # X = (X - mean) / std

    return X, y

# ---------------- Baseline 模型训练 -------------------
def train_baselines(X_train, y_train, X_test, y_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    results = {}

    # === Random Forest ===
    print("\nTraining Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,        # 增加树数量，增强模型稳定性
        max_depth=8,            # 控制单棵树复杂度，防止过拟合
        min_samples_split=10,    # 控制分裂，防止小样本暴力分裂
        min_samples_leaf=5,      # 增强泛化能力
        max_features='sqrt',     # 降低相关性提升泛化
        n_jobs=-1,
        random_state=42, 
        verbose=1
    )

    rf.fit(X_train_flat, y_train)
    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f}s")
    y_pred_rf = rf.predict(X_test_flat)
    results['rf'] = classification_report(y_test, y_pred_rf, digits=4)
    print("Random Forest:\n", results['rf'])

    # # === XGBoost ===
    # print("\nTraining XGBoost...")
    # start = time.time()
    # xgb = XGBClassifier(
    #     n_estimators=300,         # 提高容量
    #     learning_rate=0.05,       # 降低每次更新步长，减少过拟合风险
    #     max_depth=8,              # 增强模型表达力
    #     subsample=0.8,
    #     colsample_bytree=0.8,     # 用更多特征
    #     reg_alpha=1.0,            # L1正则，防止过拟合
    #     reg_lambda=1.0,           # L2正则
    #     tree_method='hist',
    #     n_jobs=-1,
    #     verbosity=1,
    #     eval_metric='logloss'
    # )

    # xgb.fit(X_train_flat, y_train)
    # elapsed = time.time() - start
    # print(f"Time taken: {elapsed:.2f}s")
    # y_pred_xgb = xgb.predict(X_test_flat)
    # results['xgb'] = classification_report(y_test, y_pred_xgb, digits=4)
    # print("XGBoost:\n", results['xgb'])

    return results

# ---------------- 主程序 -------------------
if __name__ == "__main__":
    # code_list = [
    #     # 中证A50十大权重
    #     '600519sh',  # 贵州茅台
    #     '300750sz',  # 宁德时代
    #     '601318sh',  # 中国平安
    #     '600036sh',  # 招商银行
    #     '600900sh',  # 长江电力
    #     '000333sz',  # 美的集团
    #     '002594sz',  # 比亚迪
    #     '601899sh',  # 紫金矿业
    #     '600030sh',  # 中信证券
    #     '600276sh',  # 恒瑞医药
    # ]
    
    code_list = [
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

    # train_data_folder = 'data_202111/2D_data_11-1_11-7'
    # train_data_folder_2 = 'data_202111/2D_data_11-8_11-14'
    # train_data_folder_3 = 'data_202111/2D_data_11-15_11-21'
    # test_data_folder = 'data_202111/2D_data_11-22_11-30'  

    # train_df = load_data(train_data_folder, code_list)
    # train_df_2 = load_data(train_data_folder_2, code_list)
    # train_df_3 = load_data(train_data_folder_3, code_list)
    # train_df = pd.concat([train_df, train_df_2, train_df_3], ignore_index=True)
    # X_train, y_train = preprocess_data_baseline(train_df)

    # test_df = load_data(test_data_folder, code_list, is_train=False)
    # X_test, y_test = preprocess_data_baseline(test_df, is_train=False)



    downsample_args = {
        "strategy": "fraction",
        "value": 0.2,
        "random_state": 42
    }

    train_df, test_df = load_data(
        train_data_folder_list=[
            "data2_202111/2D_data_11-1_11-7",
            "data2_202111/2D_data_11-8_11-14",
            "data2_202111/2D_data_11-15_11-21"
        ],
        test_data_folder_list=["data2_202111/2D_data_11-22_11-30"],
        code_list=code_list,
        downsample_args=downsample_args
    )

    X_train, y_train = preprocess_data_baseline(train_df)
    X_test, y_test = preprocess_data_baseline(test_df, is_train=False)


    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")

    print("训练并评估 Baseline 模型...")
    base_results = train_baselines(X_train, y_train, X_test, y_test)
    print("\n=== 模型加权 F1 对比 ===")
    print(base_results)