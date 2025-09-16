from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


REC_CNT=20
PRED_CNT=5
MAX_LEVEL=10

def proc_level2_data_with_label(df, REC_CNT=20, PRED_CNT=5, N_DAYS=3):
    """
    将高频订单簿数据转换为2D图像矩阵，并计算标签
    :param df: 输入某只股票的全部数据
    :param REC_CNT: 采样窗口大小
    :param PRED_CNT: 预测窗口大小
    :param N_DAYS: 历史波动率计算窗口天数
    :param k: 非线性压缩系数
    :param is_linear: 相对价格是否使用线性映射
    :return: DataFrame - 转换后的2D数据，每行对应一个样本图像
    """

    bid_price_cols = [f'BidPr{i}' for i in range(1, MAX_LEVEL + 1)]
    bid_volume_cols = [f'BidVol{i}' for i in range(1, MAX_LEVEL + 1)]
    ask_price_cols = [f'AskPr{i}' for i in range(1, MAX_LEVEL + 1)]
    ask_volume_cols = [f'AskVol{i}' for i in range(1, MAX_LEVEL + 1)]
    level2_cols = bid_price_cols + bid_volume_cols + ask_price_cols + ask_volume_cols

    # --- 按天预计算 sigma, 优化计算速度 ---
    df['date'] = df['datetime'].dt.floor('1D')
    all_trading_days = sorted(df['date'].unique())  # 提取所有的交易日（升序）
    real_start_date = all_trading_days[N_DAYS]
    df = df[df['datetime'] >= real_start_date].reset_index(drop=True)

    samples = []  # 样本图像数据
    labels_2 = []  # binary
    labels_3 = []  # multi-class
    for i in tqdm(range(0, len(df) - REC_CNT - PRED_CNT + 1)):
        window_df = df.loc[i:i + REC_CNT, level2_cols]  # input snapshots
        # snapshot_df = df.loc[i + REC_CNT - 1, level2_cols]
        pred_df = df.iloc[i + REC_CNT:i + REC_CNT + PRED_CNT]  # prediction horizon

        # 展平window_df
        # print(window_df.shape)
        snapshot_df = window_df.values.flatten()
        samples.append(snapshot_df)

        # 标签计算
        mid_now = (df['AskPr1'].iloc[i + REC_CNT - 1] + df['BidPr1'].iloc[i + REC_CNT - 1]) / 2
        future_mid = (pred_df['AskPr1'] + pred_df['BidPr1']) / 2
        future_mean = future_mid.mean()
        ret = (future_mean - mid_now) / mid_now

        label_2 = 1 if ret > 0 else 0
        labels_2.append(label_2)

        if ret > 0.0001:
            label_3 = 1
        elif ret < -0.0001:
            label_3 = -1
        else:
            label_3 = 0
        labels_3.append(label_3)

    col_names = [f'col_{i}' for i in range(4 * MAX_LEVEL * (REC_CNT+1))]
    df_level2 = pd.DataFrame(samples, columns=col_names)
    df_level2['label_2'] = labels_2
    df_level2['label_3'] = labels_3
    
    return df_level2


# --- 如果多进程处理函数，每只股票一个进程 ---
def process_single_stock(args):
    code, folder_path, save_path, start_date, end_date= args 
    
    if folder_path == 'data_202112' or folder_path == 'data_202111':  
        df = pd.read_csv(f'{folder_path}/{code}.csv')  # CSIA50:csv
    else:  
        df = pd.read_feather(f'{folder_path}/{code}.feather')  # CSI2000:feather

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
    print(df.shape)

    df_result = proc_level2_data_with_label(df, REC_CNT, PRED_CNT, N_DAYS=3)
    
    filename = f'{code}.feather'
    df_result.to_feather(f'{save_path}/{filename}')
  


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

for period in [
    '11-1_11-7', '11-8_11-14', '11-15_11-21', '11-22_11-30',
    '12-1_12-12', '12-13_12-19', '12-20_12-26', '12-27_12-31']:  

    if period[:2] == '11':
        folder_path = 'data2_202111'
    elif period[:2] == '12':
        folder_path = 'data2_202112'
    else:
        raise ValueError("Invalid period specified.")

    save_path = folder_path + '/snapshot_data_' + period  
    os.makedirs(save_path, exist_ok=True)


    if period == '11-1_11-7':
        start_date_str = '2021-11-01'
        end_date_str = '2021-11-08'
    elif period == '11-8_11-14':
        start_date_str = '2021-11-03'
        end_date_str = '2021-11-15'
    elif period == '11-15_11-21':
        start_date_str = '2021-11-10'
        end_date_str = '2021-11-22'
    elif period == '11-22_11-30':
        start_date_str = '2021-11-22'
        end_date_str = '2021-12-01'
    elif period == '12-1_12-12':
        start_date_str = '2021-12-01'
        end_date_str = '2021-12-13'
    elif period == '12-13_12-19':
        start_date_str = '2021-12-08'
        end_date_str = '2021-12-20'
    elif period == '12-20_12-26':
        start_date_str = '2021-12-15'
        end_date_str = '2021-12-27'
    elif period == '12-27_12-31':
        start_date_str = '2021-12-22'
        end_date_str = '2022-01-01'
    else:
        raise ValueError("Invalid period specified.")


    # 并行处理
    Parallel(n_jobs=min(len(code_list), os.cpu_count()//2), backend='loky')(
        delayed(process_single_stock)(
            (code, folder_path, save_path, start_date_str, end_date_str)
        )
        for code in code_list
    )
