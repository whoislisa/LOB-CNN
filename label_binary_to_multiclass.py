import os
import numpy as np
import pandas as pd
from tqdm import tqdm


REC_CNT=20
PRED_CNT=5

IS_BINARY=False

MAX_LEVEL = 10
SPREAD_LEVEL = 3



def calc_2D_data_with_label(df, REC_CNT=20, PRED_CNT=5, is_binary=True, N_DAYS=3, binary_data_path=None):
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

    # print("df.shape:", df.shape)
    df['date'] = df['datetime'].dt.floor('1D')
    all_trading_days = sorted(df['date'].unique())  # 提取所有的交易日（升序）
    real_start_date = all_trading_days[N_DAYS]
    df = df[df['datetime'] >= real_start_date].reset_index(drop=True)

    # print(all_trading_days, df.shape)

    labels = []

    for i in tqdm(range(0, len(df) - REC_CNT - PRED_CNT + 1)):
        # pred_df = df.iloc[i + REC_CNT:i + REC_CNT + PRED_CNT]  # prediction horizon

        # 标签计算
        mid_now = (df['AskPr1'].iloc[i + REC_CNT - 1] + df['BidPr1'].iloc[i + REC_CNT - 1]) / 2
        # future_mid = (pred_df['AskPr1'] + pred_df['BidPr1']) / 2
        future_mid = (df['AskPr1'].iloc[i + REC_CNT:i + REC_CNT + PRED_CNT] + df['BidPr1'].iloc[i + REC_CNT:i + REC_CNT + PRED_CNT]) / 2
        future_mean = future_mid.mean()
        ret = (future_mean - mid_now) / mid_now

        if is_binary:
            label = 1 if ret > 0 else 0
        else:
            if ret > 0.0001:
                label = 1
            elif ret < -0.0001:
                label = -1
            else:
                label = 0

        labels.append(label)

    df_pixels = pd.read_feather(binary_data_path)
    print(df_pixels['label'].value_counts())
    df_pixels['label'] = labels
    print(df_pixels['label'].value_counts())

    return df_pixels


# --- 多进程处理函数，每只股票一个进程 ---
def process_single_stock(args):

    code, folder_path, save_path, start_date, end_date, period = args 
    # df = pd.read_csv(f'{folder_path}/{code}.csv')
    if folder_path == 'data_202111':
        df = pd.read_csv(f'{folder_path}/{code}.csv')
    else:
        df = pd.read_feather(f'{folder_path}/{code}.feather')

    df['datetime'] = pd.to_datetime(df['datetime'])
    # start_date = '2021-11-10'  # hist: 3,4,5; cur: 8...
    # end_date = '2021-11-22'  # before: 15
    df = df[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
    print(df.shape)


    binary_data_path = f'{folder_path}/2D_data_{period}/{code}.feather'
    df_result = calc_2D_data_with_label(
        df, REC_CNT, PRED_CNT, IS_BINARY, N_DAYS=3, 
        binary_data_path=binary_data_path)
    # --- feather format ---
    filename = f'{code}.feather'
    df_result.to_feather(f'{save_path}/{filename}')
    


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

for period in ['11-1_11-7', '11-8_11-14', '11-15_11-21', '11-22_11-30']:
    save_path = 'multiclass_data_202111/2D_data_' + period  
    os.makedirs(save_path, exist_ok=True)
    for code in code_list:
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
            end_date_str = '2021-12-1'
        else:
            raise ValueError("Invalid period specified.")
        # process_single_stock((code, folder_path, save_path, start_date_str, end_date_str, period))
        # print(f"Finished processing {code}.")


folder_path = 'data2_202111'
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

for period in ['11-1_11-7', '11-8_11-14', '11-15_11-21', '11-22_11-30']:
    save_path = 'multiclass_data2_202111/2D_data_' + period  
    os.makedirs(save_path, exist_ok=True)
    for code in code_list:
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
            end_date_str = '2021-12-1'
        else:
            raise ValueError("Invalid period specified.")
        process_single_stock((code, folder_path, save_path, start_date_str, end_date_str, period))
        print(f"Finished processing {code}.")

