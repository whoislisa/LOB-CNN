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
    ],
    'backtest_data_folder_list': [
        "data_202112/test_data_12-1_12-12",
        "data_202112/test_data_12-13_12-19",
        "data_202112/test_data_12-20_12-26",
        "data_202112/test_data_12-27_12-31"
    ]
}

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
    ],
    'backtest_data_folder_list': [
        "data2_202112/test_data_12-1_12-12",
        "data2_202112/test_data_12-13_12-19",
        "data2_202112/test_data_12-20_12-26",
        "data2_202112/test_data_12-27_12-31"
    ]
}