import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from MyConfig import *


REC_CNT=20
PRED_CNT=5
MAX_LEVEL = 10
SPREAD_LEVEL = 3

def test_and_price_data_proc(df, REC_CNT=20, PRED_CNT=5, N_DAYS=3, k=3, binary_data_path=None):
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

    df['date'] = df['datetime'].dt.floor('1D')
    all_trading_days = sorted(df['date'].unique())  # 提取所有的交易日（升序）
    real_start_date = all_trading_days[N_DAYS]
    df = df[df['datetime'] >= real_start_date].reset_index(drop=True)

    infos = []  # code, time, price

    for i in tqdm(range(0, len(df) - REC_CNT - PRED_CNT + 1)):
        # pred_df = df.iloc[i + REC_CNT:i + REC_CNT + PRED_CNT]  # prediction horizon

        # 价格信息
        price_df = df.loc[i + REC_CNT, ['datetime', 'Code']].copy()
        mid_now = (df['AskPr1'].iloc[i + REC_CNT - 1] + df['BidPr1'].iloc[i + REC_CNT - 1]) / 2
        price_df['mid_price'] = mid_now

        infos.append(price_df)

    df_info = pd.DataFrame(infos, columns=['datetime', 'Code', 'mid_price'])
    df_info.reset_index(drop=True, inplace=True)
    df_pixels = pd.read_feather(binary_data_path)
    df_pixels.reset_index(drop=True, inplace=True)
    # print(df_pixels.shape)
    # print(df_info.shape)
    df_pixels = pd.concat([df_pixels, df_info], axis=1)
    # print("columns:", df_pixels.columns)
    print(f"df shape: {df_pixels.shape}")

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

    binary_data_path = f'{folder_path}/2D_data_{period}/{code}.feather'
    df_result = test_and_price_data_proc(
        df, REC_CNT, PRED_CNT, N_DAYS=3, k=k,
        binary_data_path=binary_data_path)
   
    filename = f'{code}.feather'
    df_result.to_feather(f'{save_path}/{filename}')
  

if __name__ == '__main__':
  
### CSI50  
    code_list = code_list_CSI50
    folder_path = 'data_202112'
    k = 3
    
    for period in ['12-1_12-12', '12-13_12-19', '12-20_12-26', '12-27_12-31']:  
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
        
        save_path = folder_path + '/test_data_' + period  
        os.makedirs(save_path, exist_ok=True)
        
        # 并行处理
        Parallel(n_jobs=min(len(code_list), os.cpu_count()//2), backend='loky')(
            delayed(process_single_stock)(
                (code, folder_path, save_path, start_date_str, end_date_str, k)
            )
            for code in code_list
        )



### CSI2000 
    code_list = code_list_CSI2000
    folder_path = 'data2_202112'
    k = 30
    
    for period in ['12-1_12-12', '12-13_12-19', '12-20_12-26', '12-27_12-31']:
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

        
        save_path = folder_path + '/test_data_' + period  
        os.makedirs(save_path, exist_ok=True)
        
        # 并行处理
        Parallel(n_jobs=min(len(code_list), os.cpu_count()//2), backend='loky')(
        delayed(process_single_stock)(
                (code, folder_path, save_path, start_date_str, end_date_str, k)
            )
            for code in code_list
        )

