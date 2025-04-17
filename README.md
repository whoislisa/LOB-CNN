# LOB-CNN

## 数据存储

- `data_{yyyymm}`文件夹
  - 文件夹中数据按股票代码分文件
  - 每个文件是该股票这一个月的快照数据
  - 相对于磁盘直接取出的数据，生成了`datetime`字段，并筛选了沪深两市重合的字段

- `datasets_49-stock_2d-train_3d-test`文件夹
  - `rawdata_ML/rawdata_ML.ipynb`生成的数据集，分train、test set了
