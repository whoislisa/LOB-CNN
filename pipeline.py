import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import glob
import torch

import utils as _U
import model as _M

code1 = '600176sh'
folder_path = 'data_202111/'
df1 = pd.read_csv(folder_path + code1 + '.csv')

# Chinese stock market trading time
df1 = _U.trading_time_slice(df1)

# calculate new features
df1['mid_price'] = (df1['BidPr1'] + df1['AskPr1']) / 2
pred_cnt = 5
df1['TWAP_mid'] = df1['mid_price'].rolling(window=pred_cnt).mean()
df1.dropna(axis=0, inplace=True)
df1.reset_index(drop=True, inplace=True)

# traditional ML pipeline
num_dataset = _U.generate_numerical_dataset(df1, record_cnt=5, pred_cnt=5)  # full numerical dataset
labels = [entry[1] for entry in num_dataset]  # 只取ret5标签
print(f"Label distribution: \n{pd.Series(labels).value_counts(normalize=True)}")

df_results = _U.traditional_ml_pipeline(dataset=num_dataset, balance=False, data_type='num')
print(df_results.sort_values('F1', ascending=False))

# # deep learning pipeline
# image_dataset = _U.generate_dataset(df1.iloc[:10000, :], record_cnt=5)  # full image dataset
# labels = [entry[1] for entry in image_dataset]  # 只取ret5标签
# print(f"Label distribution: \n{pd.Series(labels).value_counts(normalize=True)}")

# df_results = _U.traditional_ml_pipeline(dataset=image_dataset, balance=False, data_type='img')
# print(df_results.sort_values('F1', ascending=False))

# model = _M.BinaryCNN()
# train_loader, val_loader, test_loader = _M.create_cnn_dataloaders(image_dataset, batch_size=16, target='ret5')
# print("train_loader.dataset[0][0].shape:", train_loader.dataset[0][0].shape)

# _M.cnn_train(model, train_loader, val_loader,lr=1e-5, epochs=8, target='ret5')

# if torch.cuda.is_available():
#     torch.cuda.empty_cache()  # Clean up GPU memory
#     torch.cuda.ipc_collect()

# test_probs, test_labels = _M.cnn_predict(model, test_loader)
# metrics = _M.evaluate_cnn(test_labels, test_probs)
# df_results = pd.DataFrame({
#     'accuracy': [metrics['accuracy']],
#     'roc_auc': [metrics['roc_auc']],
#     'f1': [metrics['f1']]
# })
# df_results.to_csv('cnn_results.csv', index=False)
# print("CNN测试集结果：")
# print(df_results)
