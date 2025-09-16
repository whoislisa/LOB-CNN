import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc  # 用于释放内存
from joblib import Parallel, delayed


REC_CNT=20
PRED_CNT=5
MAX_LEVEL = 10
SPREAD_LEVEL = 3


def calc_2D_data_with_label(df, REC_CNT=20, PRED_CNT=5, N_DAYS=3, k=3):
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



    # --- 按天预计算 sigma, 优化计算速度 ---
    df['date'] = df['datetime'].dt.floor('1D')
    all_trading_days = sorted(df['date'].unique())  # 提取所有的交易日（升序）
    date_to_index = {d: i for i, d in enumerate(all_trading_days)}
    sigma_dict = {}

    for day in df['date'].unique():
        current_index = date_to_index[day]
        if current_index < N_DAYS:  # day in history
            continue
        hist_start = all_trading_days[current_index - N_DAYS]
        hist_df = df[(df['datetime'] >= hist_start) & (df['datetime'] < pd.to_datetime(day))]
        # print(hist_df['datetime'].min(), hist_df['datetime'].max())

        if len(hist_df) < 10:
            sigma_dict[day] = 0.001
        else:
            returns = hist_df[bid_price_cols[0]].pct_change(fill_method=None).dropna()
            sigma = returns.std()
            sigma_dict[day] = sigma

    # --- sigma 预计算结束 ---

    real_start_date = all_trading_days[N_DAYS]
    df = df[df['datetime'] >= real_start_date].reset_index(drop=True)
    samples = []  # 样本图像数据
    labels_2 = []  # binary
    labels_3 = []  # multi-class

    for i in tqdm(range(0, len(df) - REC_CNT - PRED_CNT + 1)):
        window_df = df.iloc[i:i + REC_CNT]  # input snapshots
        pred_df = df.iloc[i + REC_CNT:i + REC_CNT + PRED_CNT]  # prediction horizon

        p_base = window_df[bid_price_cols[0]].iloc[0]

        # 使用预计算的 sigma
        current_day = window_df['date'].iloc[0]
        sigma = sigma_dict[current_day]


        def calculate_relative_price(price, p_base):  # relative price 为纵坐标 y
            # linear / non-linear mapping
            delta_p = (price - p_base) / p_base
            if np.isnan(delta_p)or np.isnan(sigma) or sigma <= 0:
                return 112

            y = 112 * np.tanh(delta_p / (k * sigma)) + 112
            if np.isnan(y):
                return 112
            
            return np.clip(int(y), 0, 223)

        image = np.zeros((224, 3 * REC_CNT))
        V_max = window_df[bid_volume_cols + ask_volume_cols].max().max()
        for t in range(REC_CNT):
            # left col (bid side):
            for level in range(MAX_LEVEL):
                price = window_df[bid_price_cols[level]].iloc[t]
                volume = window_df[bid_volume_cols[level]].iloc[t]
                y = calculate_relative_price(price, p_base)
                if V_max <= 0 or np.isnan(V_max):
                    gray = 0
                else:
                    # gray = 255 * np.log1p(volume) / np.log1p(V_max)
                    gray = np.log1p(volume) / np.log1p(V_max)  # 归一化
                # image[y, 3 * t] = np.clip(gray, 0, 255)
                image[y, 3 * t] = np.clip(gray, 0, 1)
            
            # center col (spread):
            spread_level = SPREAD_LEVEL  # 用买卖第3档的价差
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

        label_2 = 1 if ret > 0 else 0
        labels_2.append(label_2)

        if ret > 0.0001:
            label_3 = 1
        elif ret < -0.0001:
            label_3 = -1
        else:
            label_3 = 0
        labels_3.append(label_3)

    pixel_columns = [f'pixel_{i}' for i in range(224 * 3 * REC_CNT)]
    df_pixels = pd.DataFrame(samples, columns=pixel_columns)
    df_pixels['label_2'] = labels_2
    df_pixels['label_3'] = labels_3
    
    return df_pixels


# --- 如果多进程处理函数，每只股票一个进程 ---
def process_single_stock(args):
    code, folder_path, save_path, start_date, end_date, k = args 
    
    if folder_path == 'data_202112':
        df = pd.read_csv(f'{folder_path}/{code}.csv')
    else:
        df = pd.read_feather(f'{folder_path}/{code}.feather')

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
    print(df.shape)

    df_result = calc_2D_data_with_label(df, REC_CNT, PRED_CNT, 3, k)
    
    filename = f'{code}.feather'
    df_result.to_feather(f'{save_path}/{filename}')
  
    

if __name__ == '__main__':
    
    folder_path = 'data_202112'
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

    for period in ['12-13_12-19', '12-20_12-26', '12-27_12-31']:  
        save_path = folder_path + '/2D_data_' + period  
        os.makedirs(save_path, exist_ok=True)
        
        if period == '12-1_12-12':
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
        
        k = 3
        
        # 并行处理
        Parallel(n_jobs=min(len(code_list), os.cpu_count()//2), backend='loky')(
            delayed(process_single_stock)(
                (code, folder_path, save_path, start_date_str, end_date_str, k)
            )
            for code in code_list
        )


    folder_path = 'data2_202112'
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

    for period in ['12-1_12-12', '12-13_12-19', '12-20_12-26', '12-27_12-31']:
        save_path = folder_path + '/2D_data_' + period  
        os.makedirs(save_path, exist_ok=True)
        if period == '12-1_12-12':
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

        k = 30

        # 并行处理
        Parallel(n_jobs=min(len(code_list), os.cpu_count()//2), backend='loky')(
        delayed(process_single_stock)(
                (code, folder_path, save_path, start_date_str, end_date_str, k)
            )
            for code in code_list
        )

