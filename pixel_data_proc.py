import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc  # 用于释放内存
from joblib import Parallel, delayed


REC_CNT=20
PRED_CNT=5
IS_LINEAR=False
IS_BINARY=True
IS_TRAIN=True

MAX_LEVEL = 10
SPREAD_LEVEL = 3


def calc_2D_data_with_label(df, REC_CNT=20, PRED_CNT=5, is_binary=True, is_linear=False, N_DAYS=3, k=3):
    """
    将高频订单簿数据转换为2D图像矩阵，并计算标签
    :param df: 输入某只股票的全部数据
    :param REC_CNT: 采样窗口大小
    :param PRED_CNT: 预测窗口大小
    :param N_DAYS: 历史波动率计算窗口天数
    :param k: 非线性压缩系数
    :param is_linear: 相对价格是否使用线性映射
    :param is_binary: 标签是否使用二分类
    :return: DataFrame - 转换后的2D数据，每行对应一个样本图像
    """

    bid_price_cols = [f'BidPr{i}' for i in range(1, MAX_LEVEL + 1)]
    bid_volume_cols = [f'BidVol{i}' for i in range(1, MAX_LEVEL + 1)]
    ask_price_cols = [f'AskPr{i}' for i in range(1, MAX_LEVEL + 1)]
    ask_volume_cols = [f'AskVol{i}' for i in range(1, MAX_LEVEL + 1)]



    # --- 按天预计算 sigma, L, U, 优化计算速度 ---
    df['date'] = df['datetime'].dt.floor('1D')
    all_trading_days = sorted(df['date'].unique())  # 提取所有的交易日（升序）
    date_to_index = {d: i for i, d in enumerate(all_trading_days)}
    # print("date_to_index:", date_to_index)
    sigma_dict = {}
    LU_dict = {}

    for day in df['date'].unique():
        # hist_start = pd.to_datetime(day) - pd.Timedelta(days=N_DAYS)
        current_index = date_to_index[day]
        if current_index < N_DAYS:  # day in history
            continue
        hist_start = all_trading_days[current_index - N_DAYS]
        hist_df = df[(df['datetime'] >= hist_start) & (df['datetime'] < pd.to_datetime(day))]
        # print(hist_df['datetime'].min(), hist_df['datetime'].max())

        if len(hist_df) < 10:
            sigma_dict[day] = 0.001
            LU_dict[day] = (-0.01, 0.01)
        else:
            # returns = hist_df[bid_price_cols[0]].pct_change().dropna()
            returns = hist_df[bid_price_cols[0]].pct_change(fill_method=None).dropna()
            sigma = returns.std()
            sigma_dict[day] = sigma

            p_base = hist_df[bid_price_cols[0]].iloc[0]
            delta_p = (hist_df[bid_price_cols].values - p_base) / p_base
            L, U = np.nanquantile(delta_p, [0.05, 0.95]) if len(delta_p) else (-0.01, 0.01)
            if np.isnan(delta_p).all():
                L, U = (-0.01, 0.01)
            LU_dict[day] = (L, U)
    # --- sigma, L, U 预计算结束 ---

    real_start_date = all_trading_days[N_DAYS]
    df = df[df['datetime'] >= real_start_date].reset_index(drop=True)
    samples = []  # 样本图像数据
    labels = []

    for i in tqdm(range(0, len(df) - REC_CNT - PRED_CNT + 1)):
        window_df = df.iloc[i:i + REC_CNT]  # input snapshots
        pred_df = df.iloc[i + REC_CNT:i + REC_CNT + PRED_CNT]  # prediction horizon

        p_base = window_df[bid_price_cols[0]].iloc[0]

        # 使用预计算的 sigma, L, U
        current_day = window_df['date'].iloc[0]
        sigma = sigma_dict[current_day]
        L, U = LU_dict[current_day]

        def calculate_relative_price(price, p_base):  # relative price 为纵坐标 y
            # linear / non-linear mapping
            delta_p = (price - p_base) / p_base
            if np.isnan(delta_p):
                return 112
            if is_linear:
                y = int(224 * (delta_p - L) / (U - L))
            else:
                y = int(112 * np.tanh(delta_p / (k * sigma)) + 112)
            
            if np.isnan(y):
                return 112
            return np.clip(y, 0, 223)

        image = np.zeros((224, 3 * REC_CNT))
        for t in range(REC_CNT):
            # left col (bid side):
            for level in range(MAX_LEVEL):
                price = window_df[bid_price_cols[level]].iloc[t]
                volume = window_df[bid_volume_cols[level]].iloc[t]
                y = calculate_relative_price(price, p_base)
                V_max = window_df[bid_volume_cols + ask_volume_cols].max().max()
                if V_max <= 0 or np.isnan(V_max):
                    gray = 0
                else:
                    # gray = 255 * np.log1p(volume) / np.log1p(V_max)
                    gray = np.log1p(volume) / np.log1p(V_max)  # 归一化
                # image[y, 3 * t] = np.clip(gray, 0, 255)
                image[y, 3 * t] = np.clip(gray, 0, 1)
            
            # center col (spread):
            spread_level = 3  # 用买卖第3档的价差
            spread_price = []
            for side in ['Bid', 'Ask']:
                price = window_df[f'{side}Pr{spread_level}'].iloc[t]
                y = calculate_relative_price(price, p_base)
                spread_price.append(y)
            # image[spread_price[0]:spread_price[1], 3 * t + 1] = 255
            image[spread_price[0]:spread_price[1], 3 * t + 1] = 1

            # right col (ask side):
            for level in range(MAX_LEVEL):
                price = window_df[ask_price_cols[level]].iloc[t]
                volume = window_df[ask_volume_cols[level]].iloc[t]
                y = calculate_relative_price(price, p_base)
                V_max = window_df[bid_volume_cols + ask_volume_cols].max().max()
                if V_max <= 0 or np.isnan(V_max):
                    gray = 0
                else:
                    # gray = 255 * np.log1p(volume) / np.log1p(V_max)
                    gray = np.log1p(volume) / np.log1p(V_max)  # 归一化
                # image[y, 3 * t + 2] = np.clip(gray, 0, 255)
                image[y, 3 * t + 2] = np.clip(gray, 0, 1)

        samples.append(image.flatten())

        # 标签计算
        mid_now = (df['AskPr1'].iloc[i + REC_CNT - 1] + df['BidPr1'].iloc[i + REC_CNT - 1]) / 2
        future_mid = (pred_df['AskPr1'] + pred_df['BidPr1']) / 2
        future_mean = future_mid.mean()
        ret = (future_mean - mid_now) / mid_now

        if is_binary:
            label = 1 if ret > 0 else 0
        else:
            if ret > 0.001:
                label = 1
            elif ret < -0.001:
                label = -1
            else:
                label = 0

        labels.append(label)

    pixel_columns = [f'pixel_{i}' for i in range(224 * 3 * REC_CNT)]
    df_pixels = pd.DataFrame(samples, columns=pixel_columns)
    df_pixels['label'] = labels
    return df_pixels


# --- 多进程处理函数，每只股票一个进程 ---
def process_single_stock(args):
    try:
        code, folder_path, save_path = args 
        df = pd.read_csv(f'{folder_path}/{code}.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        start_date = '2021-11-10'  # hist: 3,4,5; cur: 8...
        end_date = '2021-11-22'  # before: 15
        df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
        print(df.shape)

        df_result = calc_2D_data_with_label(df, REC_CNT, PRED_CNT, IS_BINARY, IS_LINEAR, 3, 3)
        if not IS_TRAIN:
            # save_path = f'{save_path}/test'
            pass

        # --- feather format ---
        filename = f'{code}.feather'
        df_result.to_feather(f'{save_path}/{filename}')
    
    except Exception as e:
        print(f"[ERROR] {args[1]}/{args[0]}.csv 处理失败：{e}")

def process_single_stock_2(args):
    try:
        code, folder_path, save_path = args 
        df = pd.read_csv(f'{folder_path}/{code}.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        start_date = '2021-11-22'  # hist: 3,4,5; cur: 8...
        end_date = '2021-12-1'  # before: 15
        df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
        print(df.shape)

        df_result = calc_2D_data_with_label(df, REC_CNT, PRED_CNT, IS_BINARY, IS_LINEAR, 3, 3)
        if not IS_TRAIN:
            # save_path = f'{save_path}/test'
            pass

        # --- feather format ---
        filename = f'{code}.feather'
        df_result.to_feather(f'{save_path}/{filename}')
    
    except Exception as e:
        print(f"[ERROR] {args[1]}/{args[0]}.csv 处理失败：{e}")
    

if __name__ == '__main__':
    folder_path = 'data_202111'
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

    save_path = 'data_202111/2D_data_11-15_11-21_10level'
    os.makedirs(save_path, exist_ok=True)
    # 并行处理  n_jobs=min(len(code_list), os.cpu_count()//2); os.cpu_count() = 8
    Parallel(n_jobs=3, backend='loky')(
    delayed(process_single_stock)((code, folder_path, save_path))
    for code in code_list[5:]
    )

    save_path = 'data_202111/2D_data_11-22_11-30_10level'
    os.makedirs(save_path, exist_ok=True)
    # 并行处理  n_jobs=min(len(code_list), os.cpu_count()//2); os.cpu_count() = 8
    Parallel(n_jobs=4, backend='loky')(
    delayed(process_single_stock_2)((code, folder_path, save_path))
    for code in code_list
    )

