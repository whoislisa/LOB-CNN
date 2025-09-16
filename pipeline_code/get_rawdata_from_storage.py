import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import time
from datetime import datetime
import glob

import utils as _U


### used_cols
level2_col_list = []
for l in range(1, 11):
    BidPr_col = f'BidPr{l}'
    BidVol_col = f'BidVol{l}'
    AskPr_col = f'AskPr{l}'
    AskVol_col = f'AskVol{l}'
    level2_col_list.extend([BidPr_col, BidVol_col, AskPr_col, AskVol_col])
used_cols = [
    'datetime', 'Exchflg', 'Code', 'Code_Mkt', 'Qdate', 'QTime', 'InstrumentStatus', 'Trdirec',
    'PrevClPr', 'OpPr', 'HiPr', 'LoPr', 'Tprice', 'Tvolume', 'Tsum', 'Tdeals', 'TVolume_accu', 'TSum_accu', 'Tdeals_accu',
    'TotBidVol', 'WghtAvgBidPr', 'TotAskVol', 'WghtAvgAskPr',
    'Absspread', 'Respread', 'Abseffspread', 'Reeffspread', 'Depth1', 'Depth2'
]
used_cols.extend(level2_col_list)


### code_list
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
    # '002611sz',  # 东方精工
    # '300607sz',  # 拓斯达
    # '300010sz',  # 豆神教育
    # '300007sz',  # 汉威科技
    # '688158sh',  # 优刻得
    # '300493sz',  # 润欣科技
    # '600410sh',  # 华胜天成
    # '002073sz',  # 软控股份
    # '603887sh',  # 城地香江
    # '300098sz',  # 高新兴
]


# 获取 ~/ 目录下所有 snap_stkhf202112_*.sas7bdat 文件
year_month = '202112'
for code in code_list:
    file_path = glob.glob(f'/mnt/sdb1/HF{year_month[:4]}/L2HF{year_month[2:]}_L2/snap_stkhf{year_month}_{code}.sas7bdat')
    if file_path:
        train_df = pd.read_sas(file_path[0], format='sas7bdat', encoding='utf-8')
        train_df.to_csv(f'rawdata_L2HF{year_month[2:]}_L2/{code}.csv', index=False)
        
        # convert all df in dfs to datetime
        date = pd.to_datetime(train_df['Qdate'])
        time = pd.to_timedelta(train_df['QTime'], unit='s')
        train_df['datetime'] = date + time
        
        # select used cols
        train_df = train_df[used_cols]
        train_df = _U.trading_time_slice(train_df)
        train_df.to_feather(f'data2_{year_month}/{code}.feather')
        
        print(f'{code}.feather saved. Shape: {train_df.shape}')
    else:
        print(f'{code} not found')  # 上市日期若晚于2021年11月




