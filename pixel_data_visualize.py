# 用于数据可视化
# 以及验证数据处理是否有效


import pandas as pd
import matplotlib.pyplot as plt

def visualize_image_sample(flat_sample, REC_CNT=20):
    """
    将flatten后的图像样本可视化（224 x 3*REC_CNT）
    """
    image = flat_sample.values.reshape(224, 3*REC_CNT)
    # times every entry by 255 to convert to 0-255 range
    image = image * 255
    plt.figure(figsize=(12, 6))
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.title("Order Book Image Sample")
    plt.xlabel("Time")
    plt.ylabel("Price Axis")
    plt.colorbar(label="Normalized Volume (Gray Scale)")
    plt.show()
    # save

# 可视化第一个样本
REC_CNT = 20
pix_df = pd.read_feather('data2_202111/2D_data_11-1_11-7/300718sz.feather') 
visualize_image_sample(pix_df.iloc[2060, :-1], REC_CNT=REC_CNT)
