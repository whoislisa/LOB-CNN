import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC  
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import time


# Chinese stock market trading time
START_TRADE_T1 = pd.to_timedelta('09:30:00').total_seconds()
END_TRADE_T1 = pd.to_timedelta('11:30:00').total_seconds()
START_TRADE_T2 = pd.to_timedelta('13:00:00').total_seconds()
END_TRADE_T2 = pd.to_timedelta('14:57:00').total_seconds()

def trading_time_slice(df: pd.DataFrame):
    trading_time_mask = (
    ((df['QTime'] > START_TRADE_T1) & (df['QTime'] < END_TRADE_T1)) |
    ((df['QTime'] > START_TRADE_T2) & (df['QTime'] < END_TRADE_T2))
)
    df = df[trading_time_mask]
    df.reset_index(drop=True, inplace=True)
    df.loc[:, 'datetime'] = pd.to_datetime(df['datetime'])
    
    return df

# Price transformation according to 中科大Paper
PLOT_I = 2
PLOT_M = 10
PLOT_K = 10

def transformed_price(price: float, best_bid: float, tick_size=0.01) -> int:
    """
    将价格转换为相对于最佳买价的相对价格，然后缩放
    :param price: 价格
    :param best_bid: 最佳买价
    :return: 相对价格
    """
    rel_price = price - best_bid
    rel_price /= tick_size
    
    if rel_price >= - PLOT_M and rel_price <= PLOT_M:
        return round(rel_price)
    elif rel_price > PLOT_M:
        return min(20, round((rel_price - 1) // PLOT_K + PLOT_M)) 
    else:
        return max(-20, round(- ((abs(rel_price) - 1) // PLOT_K + PLOT_M)))

def single_image(snapshot_df: pd.DataFrame, record_cnt=5, pred_cnt=5, is_binary=True) -> list:
    """
    生成单张图像
    :param snapshot_df: 包含 Level 2 数据的 DataFrame, shape[0] > record_cnt
    :param record_cnt: 记录数量
    :return: list in the form of [np.array(image_size), binary, binary]. The binaries (0./1.) are the label of ret5, ret30
    
    Note: A single record's data occupies 3 pixels (width)
    
    """
    assert snapshot_df.shape[0] > record_cnt, "Error: expected snapshot_df.shape[0] > record_cnt"
    
    image_size = (41, 3*record_cnt)
    image = np.zeros(image_size)
    # current best bid and mid price
    snapshot_df.reset_index(drop=True, inplace=True)
    best_bid = snapshot_df.loc[record_cnt-1, 'BidPr1']
    
    for i in range(record_cnt):
        snapshot = snapshot_df.iloc[i]
        # plot_snapshot(snapshot)
        X_shift = 3*i
        
        ask_trans_p_list = []
        bid_trans_p_list = []
        ask_v_list = []
        bid_v_list = []
        for i in range(10):
            ask_trans_p_list.append(transformed_price(snapshot[f'AskPr{i+1}'], best_bid))
            bid_trans_p_list.append(transformed_price(snapshot[f'BidPr{i+1}'], best_bid))
            ask_v_list.append(snapshot[f'AskVol{i+1}'])
            bid_v_list.append(snapshot[f'BidVol{i+1}'])

        # draw price line: level-5
        image[20+bid_trans_p_list[4]:20+ask_trans_p_list[4], X_shift+1] = 255
        # draw bid/ask levels: level-10
        v_max = max(max(ask_v_list), max(bid_v_list))
        for i in range(10):
            image[20+bid_trans_p_list[i], X_shift] += bid_v_list[i] / v_max * 255
            image[20+ask_trans_p_list[i], X_shift+2] += ask_v_list[i] / v_max * 255

        # for any pixel in image, if it's value is larger than 255, set it to 255
        image[image > 255] = 255
    
    def calc_label(snapshot_df, record_cnt, pred_cnt, is_binary=True):
        diff = snapshot_df.loc[record_cnt - 1 + pred_cnt, 'TWAP_mid'] - snapshot_df.loc[record_cnt - 1, 'mid_price']
        diff = round(diff, 2)
        if is_binary:
            return 1 if diff >= 0.01 else 0
        else:
            return 1 if diff >= 0.01 else -1 if diff <= -0.01 else 0
    label = calc_label(snapshot_df, record_cnt, pred_cnt, is_binary)

    return [image, label]

def display_image(entry):
    """
    Display image of a single entry
    :param entry: list in the form of [np.array(image_size), binary, binary]
    """
    assert (type(entry) == list) and (len(entry) == 3), "Type error, expected a list with length of 3"
    plt.figure
    plt.imshow(entry[0], cmap=plt.get_cmap('gray'))
    plt.ylim((0,entry[0].shape[0]-1))
    plt.xlim((0,entry[0].shape[1]-1))
    plt.title(f'ret5: {entry[2]}\nret20: {entry[2]}')
    plt.show()
   
def plot_snapshot(snapshot: pd.Series):
    """
    绘制某一时刻的 Level 2 数据 histogram
    :param snapshot: 包含 Level 2 数据的 Series
    """
    bid_prices = snapshot[[f'BidPr{i}' for i in range(1, 11)]].values
    bid_volumes = snapshot[[f'BidVol{i}' for i in range(1, 11)]].values
    ask_prices = snapshot[[f'AskPr{i}' for i in range(1, 11)]].values
    ask_volumes = snapshot[[f'AskVol{i}' for i in range(1, 11)]].values
    
    # 绘制 Bid 和 Ask 价格
    plt.figure(figsize=(4, 2))
    # use histograms
    plt.bar(bid_prices, bid_volumes, width=0.01, color='b', alpha=0.7, label='Bid Prices')
    plt.bar(ask_prices, ask_volumes, width=0.01, color='r', alpha=0.7, label='Ask Prices')
    plt.xlabel('Price')
    plt.ylabel('Volume')
    plt.title(f'Level 2 Data at Time {snapshot["datetime"]}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def generate_dataset(df, record_cnt=5, pred_cnt=5) -> list:
    '''Generate dataset of entries from df'''
    dataset = []
    for i in tqdm(range(len(df) - (record_cnt + 30))):
        entry = single_image(df.iloc[i:i + (record_cnt + 30)], record_cnt, pred_cnt)
        dataset.append(entry)
    return dataset

def calc_label(snapshot_df, record_cnt, offset, mid_price, is_binary=True):
    """
    :param snapshot_df: DataFrame containing Level 2 data, shape[0] >= record_cnt + offset
    :param record_cnt: Number of records
    :param offset: Offset for return calculation
    :param mid_price: Mid price at the time of the last record in graph
    """
    df = snapshot_df.copy()
    df.reset_index(drop=True, inplace=True)
    diff = df.loc[record_cnt - 1 + offset, 'mid_price'] - mid_price
    diff = round(diff, 2)
    if is_binary:  # binary
        return 1 if diff >= 0.01 else 0
    else:  # multi-class
        return 1 if diff >= 0.01 else -1 if diff <= -0.01 else 0

def generate_numerical_dataset(df, record_cnt=5, pred_cnt=5, is_binary=True) -> list:
    '''
    Generate dataset of numerical vol + price info
    :param df: DataFrame containing Level 2 data
    :param record_cnt: Number of records
    :return: dataset of numerical data
    :rtype: list of lists, each list contains [input, label]
    '''
    feature_list = []
    for l in range(1, 11):
        feature_list.append(f'BidPr{l}')
        feature_list.append(f'BidVol{l}')
        feature_list.append(f'AskPr{l}')
        feature_list.append(f'AskVol{l}')
    feature_list.append('mid_price')
    
    dataset = []
    for i in tqdm(range(len(df) - (record_cnt + pred_cnt))):
        # input: numerical 
        input = df.iloc[i:i + record_cnt][feature_list].values  # (35, 40), flattened before training
        # labels
        snapshot_df = df.iloc[i:i + record_cnt + pred_cnt, :]
        mid_price = snapshot_df.loc[i + record_cnt - 1, 'mid_price']
        label = calc_label(snapshot_df, record_cnt, pred_cnt, mid_price, is_binary)

        dataset.append([input, label])
    return dataset

def save_report(df: pd.DataFrame, model_name='', balance=False, task_type='binary'):
    '''Save report to csv file'''
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    balance_tag = "balanced" if balance else "unbalanced"
    fname = f"reports/{task_type}_{model_name}_{balance_tag}_{timestamp}.csv"
    df.to_csv(fname, index=True)
    print(f"Report saved to: {fname}")

def traditional_ml_pipeline(dataset: dict, balance=False, data_type='img'):
    '''
    多个传统机器学习模型训练和评估
    :param dataset: 
    :param balance: 是否进行类别平衡处理
    :param data_type: 数据类型，'img' 或 'num'
    :return: 训练和评估结果的 DataFrame
    
    dataset: list of lists, each list contains [input, label]
    '''
    # 数据准备
    train_X_flat = dataset['train_X']
    train_y = dataset['train_y']
    test_X_flat = dataset['test_X']
    test_y = dataset['test_y']
    
    if data_type == 'img':  # 如果是图像数据需要展平
        train_X_flat = train_X_flat.reshape(train_X_flat.shape[0], -1)
        test_X_flat = test_X_flat.reshape(test_X_flat.shape[0], -1)

    # 标准化处理（仅使用训练集统计量）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_X_flat)
    X_test_scaled = scaler.transform(test_X_flat)

    # 类别平衡处理（仅训练集）
    if balance:
        smote = SMOTE(random_state=42)
        X_train_scaled, train_y = smote.fit_resample(X_train_scaled, train_y)
    
    # 模型列表
    models = [
        ('Logistic Regression (SGD)',
        SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=1e-4,          # 正则化参数
            max_iter=1000, 
            tol=1e-3,
            class_weight='balanced' if balance else None,
            n_jobs=-1,              # 并行计算
            random_state=42
        )),

        ('Linear SVM',
        LinearSVC(
            C=1.0,                  # 正则化参数
            class_weight='balanced' if balance else None,
            dual=False,             # 避免大数据集的求解问题
            max_iter=2000,          # 迭代次数增加以防止收敛失败
            tol=1e-4,
            random_state=42
        )),

        ('XGBoost',
        XGBClassifier(
            objective='binary:logistic',  # 二分类问题
            n_estimators=500,
            # early_stopping_rounds=50,  # 早停
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            tree_method='hist',           # 适用于中等数据
            n_jobs=-1,                    # 并行加速
            random_state=42
        )),

        ('MLP',
        MLPClassifier(
            hidden_layer_sizes=(128, 64), # 两层隐藏层，神经元数 128 → 64
            activation='relu',            # ReLU 激活函数
            solver='adam',                # Adam 优化
            alpha=1e-4,                   # L2 正则化
            batch_size=128,               # 小批量梯度下降
            learning_rate_init=0.001,      # 学习率
            max_iter=500,                 # 训练 500 轮
            early_stopping=True,          # 提前停止，防止过拟合
            n_iter_no_change=10,          # 10 轮无提升则停止
            random_state=42
        ))

    ]
    
    # 训练评估
    results = []
    for name, model in models:
        start_time = time.time()
        
        model.fit(X_train_scaled, train_y)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(test_y, y_pred, output_dict=True)
        
        elapsed = time.time() - start_time
        result = {
            'Model': name,
            'Accuracy': report['accuracy'],
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1': report['1']['f1-score']
        }
        results.append(result)
        print(f"--- {name} ---")
        print("Time elapsed: ", elapsed, "(s)")
        report_df = pd.DataFrame(report).transpose()
        print(report_df)
        
    result_df = pd.DataFrame(results)
    result_df.set_index('Model', inplace=True)
    result_df.sort_values('F1', ascending=False, inplace=True)
    save_report(result_df.sort_values('F1', ascending=False), model_name='Traditional', balance=balance, task_type='binary')
    return pd.DataFrame(results)
