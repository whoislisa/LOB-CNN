# LOB-CNN

## 流程

- 在远程服务器运行 `get_rawdata_from_storage.py` 从磁盘获取原始快照数据，并做初步处理（时间戳，沪、深共用的特征）
- 本地运行 `pixel_data_proc.py` 将原始快照数据处理成“灰度图”形式，完成归一化，适用于简单机器学习模型和深度学习模型输入
- 可以通过本地运行 `pixel_data_visualize.py` 可视化并检验上一步中数据转换的效果
- 运行 `run_baseline_model.py` 对数据进行降采样，然后在简单机器学习模型（如线性模型、随机森林）上训练和测试
- 运行 `run_cnn_model.py` 对数据进行降采样，然后在深度学习模型上训练和测试
- 或者运行 `run2_cnn_model.py`用于训练模型，用了两个月的数据，逻辑同上；运行 `test_cnn.py` 测试已保存的模型


## 数据生成

- 数据在本地生成，因为本地 CPU 比服务器上的快
- 数据生成完毕，将文件生成压缩包，本地终端 scp 到服务器，如 `scp "/Users/liuyufei/Desktop/2D_data_11-22_11-30.zip" qbzhou21@166.111.96.24:/home/qbzhou21/liuyufei/local_data` 
- 在服务器终端 cd 到对应文件夹，`unzip /home/qbzhou21/liuyufei/local_data/2D_data_11-22_11-30.zip` 解压数据到当前文件夹


## 数据存储

- `data_{yyyymm}`文件夹
  - 中证A50成分股，权重top10
  - 信息来源：https://www.csindex.com.cn/#/
  - 文件夹中数据按股票代码分文件
  - 每个文件是该股票这一个月的快照数据
  - 相对于磁盘直接取出的数据，生成了`datetime`字段，并筛选了沪深两市重合的字段
    ```
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
    ```

- `data2_{yyyymm}`文件夹
  - 中证2000成分股，权重top10
  - 处理方式同上
    ```
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
    
    ```

- `datasets_49-stock_2d-train_3d-test`文件夹
  - `rawdata_ML/rawdata_ML.ipynb`生成的数据集，分train、test set了


