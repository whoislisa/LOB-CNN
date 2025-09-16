# ML

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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import joblib 


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
    for folder in data_folder_list:           
        for code in code_list:
            path = os.path.join(folder, f"{code}.feather")
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            df = pd.read_feather(path)
            if downsample_args:
                df = sample_df(df, downsample_args)
            if is_Nov:
                X = df.drop(columns=['label']).values
                y = df['label'].values
            else:
                X = df.drop(columns=['label_2', 'label_3']).values
                y = df['label_2'].values
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


# ---------------- 数据预处理 -------------------
def preprocess_data(df, balanced=False, random_state=42):
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # 【新增】支持是否平衡正负样本
    if balanced:
        X_pos = X[y == 1]
        X_neg = X[y == 0]
        n_samples = min(len(X_pos), len(X_neg))
        X_pos_down = resample(X_pos, replace=False, n_samples=n_samples, random_state=random_state)
        X_neg_down = resample(X_neg, replace=False, n_samples=n_samples, random_state=random_state)
        X = np.vstack([X_pos_down, X_neg_down])
        y = np.array([1]*n_samples + [0]*n_samples)
        # 打乱
        shuffle_idx = np.random.permutation(len(y))
        X, y = X[shuffle_idx], y[shuffle_idx]

    return X, y

# ---------------- 训练传统机器学习模型 -------------------
# def train_trad_ML(X_train, y_train):
#     # 【修正】补全训练多个模型，并返回列表
#     from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.svm import SVC

#     model_list = []

#     # 随机森林
#     rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     model_list.append(('RandomForest', rf))

#     # GBDT
#     gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#     gbdt.fit(X_train, y_train)
#     model_list.append(('GBDT', gbdt))

#     # 逻辑回归
#     lr = LogisticRegression(max_iter=1000, random_state=42)
#     lr.fit(X_train, y_train)
#     model_list.append(('LogisticRegression', lr))

#     # SVM（如果样本量太大可以注释）
#     # svc = SVC(kernel='linear', probability=True, random_state=42)
#     # svc.fit(X_train, y_train)
#     # model_list.append(('SVM', svc))

#     # 保存模型
#     os.makedirs('saved_models', exist_ok=True)
#     for name, model in model_list:
#         joblib.dump(model, f'saved_models/{name}.pkl')
#         print(f"Saved model {name}.")

#     return model_list


# ---------------- 训练传统机器学习模型 -------------------
def train_trad_ML(X_train, y_train):
    import os
    import joblib
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    import xgboost as xgb

    model_list = []

    os.makedirs('saved_models', exist_ok=True)

    # # 随机森林（CPU并行）
    # print("Training RandomForest...")
    # start_time = time.time()
    # rf = RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=15,
    #     min_samples_split=10,
    #     n_jobs=-1,
    #     verbose=1,
    #     random_state=42
    # )
    # rf.fit(X_train, y_train)
    # model_list.append(('RandomForest', rf))
    # print(f"RandomForest training done. Time: {time.time() - start_time:.2f} sec")

    # LightGBM（可以用GPU）
    print("\nTraining LightGBM...")
    start_time = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        device='cpu',  # GPU加速
        random_state=42,
        verbose=50,
        n_jobs=-1
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train)])
    model_list.append(('LightGBM', lgbm))
    print(f"LightGBM training done. Time: {time.time() - start_time:.2f} sec")

    # XGBoost（可以用GPU）
    print("\nTraining XGBoost...")
    start_time = time.time()
    xgbm = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        tree_method='gpu_hist',  # GPU加速
        predictor='gpu_predictor',
        max_bin=256,  # 降低显存占用，防止OOM
        random_state=42,
        verbosity=1,
        n_jobs=-1
    )
    xgbm.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=50)
    model_list.append(('XGBoost', xgbm))
    print(f"XGBoost training done. Time: {time.time() - start_time:.2f} sec")

    # # 逻辑回归（CPU，不太适合840特征，但可做baseline）
    # print("\nTraining LogisticRegression...")
    # start_time = time.time()
    # lr = LogisticRegression(
    #     solver='saga',
    #     penalty='l2',
    #     max_iter=1000,
    #     n_jobs=-1,
    #     verbose=1,
    #     random_state=42
    # )
    # lr.fit(X_train, y_train)
    # model_list.append(('LogisticRegression', lr))
    # print(f"LogisticRegression training done. Time: {time.time() - start_time:.2f} sec")

    # 保存模型
    for name, model in model_list:
        joblib.dump(model, f'saved_models/{name}.pkl')
        print(f"Saved model {name}.")

    return model_list

 
# ---------------- 预测并保存评估结果 -------------------
def evaluate_and_save(y_true, y_pred, model_path, save_path=None, tag=""):
    print(f"评估 {tag} ...")
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    if save_path:
        report_df.to_csv(f"{save_path}_{tag}_classification_report.csv", index=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {tag}")
    if save_path:
        plt.savefig(f"{save_path}_{tag}_confusion_matrix.png")
    plt.close()

def predict_and_collect(model_path, save_path, X_test, y_test, week_idx):

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    all_preds = list(y_pred)
    all_labels = list(y_test)

    # 每周单独保存指标
    evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=model_path, save_path=save_path, tag=f"week{week_idx+1}")

    return all_preds, all_labels

# ---------------- 主流程 -------------------


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



data_args = data_args_CSI50

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

X_train, y_train = preprocess_data(train_df, balanced=False)    
print(f"训练集大小: {X_train.shape}")
print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")

print("开始训练")
cur_time = datetime.now().strftime("%m%d-%H%M")
name_str = f"M_trad_D_{data_args['name']}_T_{cur_time}"
print(f"保存路径: {name_str}")

model_list = train_trad_ML(X_train, y_train)

print("开始评估")
for model_name, model in model_list:

    all_preds = []
    all_labels = []

    model_path = f'saved_models/{model_name}.pkl'
    save_path = f"trad_ML_results_on_2D_data/{name_str}_{model_name}"
    os.makedirs("trad_ML_results_on_2D_data", exist_ok=True)

    for week_idx, week_folder in enumerate(data_args['test_data_folder_list']):
        print(f"加载第 {week_idx+1} 周的数据...")
        test_df = load_data(
            data_folder_list=[week_folder],  # 注意这里是 [week_folder]，每次一个
            code_list=data_args['code_list'],
        )
        X_test, y_test = preprocess_data(test_df)
        print(f"本周测试集大小: {X_test.shape}")
        print(f"本周标签分布: {np.unique(y_test, return_counts=True)}")

        preds, labels = predict_and_collect(model_path, save_path, X_test, y_test, week_idx)
        all_preds.extend(preds)
        all_labels.extend(labels)

    print("整体评估（全月数据）...")
    evaluate_and_save(np.array(all_labels), np.array(all_preds), model_path=model_path, save_path=save_path, tag="overall")

