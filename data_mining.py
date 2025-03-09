import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# 模擬資料 (實際使用時請替換成你的資料)
residual_data = torch.randn(1, 96, 64, 120)
mv_data = torch.randn(1, 128, 64, 120)
z_data = torch.randn(1, 64, 16, 30)

def analyze_data(data, name="data"):
    """
    分析資料的整體統計特性，並繪製直方圖和盒鬚圖。
    """
    # 將資料展平成一維陣列
    data_np = data.cpu().numpy().flatten()
    
    # 計算統計量
    mean_val = np.mean(data_np)
    std_val = np.std(data_np)
    min_val = np.min(data_np)
    max_val = np.max(data_np)
    median_val = np.median(data_np)
    skew_val = skew(data_np)
    kurt_val = kurtosis(data_np)
    
    print(f"==== {name} 統計量 ====")
    print(f"平均值: {mean_val:.4f}")
    print(f"標準差: {std_val:.4f}")
    print(f"最小值: {min_val:.4f}")
    print(f"最大值: {max_val:.4f}")
    print(f"中位數: {median_val:.4f}")
    print(f"偏度: {skew_val:.4f}")
    print(f"峰度: {kurt_val:.4f}")
    print("\n")
    
    # 繪製直方圖
    plt.figure(figsize=(8, 4))
    plt.hist(data_np, bins=50, alpha=0.7, color='blue')
    plt.title(f"{name} 的直方圖")
    plt.xlabel("數值")
    plt.ylabel("頻次")
    plt.show()
    
    # 繪製盒鬚圖
    plt.figure(figsize=(6, 4))
    plt.boxplot(data_np, vert=False)
    plt.title(f"{name} 的盒鬚圖")
    plt.xlabel("數值")
    plt.show()

def analyze_channels(data, name="data"):
    """
    分析每個通道的統計特性，並繪製各通道的平均值和標準差散點圖。
    假設資料形狀為 [1, C, H, W]。
    """
    data_np = data.cpu().numpy()
    channels = data_np.shape[1]
    channel_means = []
    channel_stds = []
    
    for c in range(channels):
        channel_data = data_np[0, c, :, :].flatten()
        channel_means.append(np.mean(channel_data))
        channel_stds.append(np.std(channel_data))
    
    # 繪製通道平均值
    plt.figure(figsize=(8, 4))
    plt.scatter(range(channels), channel_means, s=40)
    plt.title(f"{name} 各通道平均值")
    plt.xlabel("通道編號")
    plt.ylabel("平均值")
    plt.grid(True)
    plt.show()
    
    # 繪製通道標準差
    plt.figure(figsize=(8, 4))
    plt.scatter(range(channels), channel_stds, s=40, color='orange')
    plt.title(f"{name} 各通道標準差")
    plt.xlabel("通道編號")
    plt.ylabel("標準差")
    plt.grid(True)
    plt.show()

def visualize_first_channel(data, name="data"):
    """
    視覺化資料第一個通道的內容 (假設資料為 [1, C, H, W])。
    """
    data_np = data.cpu().numpy()[0, 0, :, :]
    plt.figure(figsize=(6,5))
    plt.imshow(data_np, cmap='viridis')
    plt.title(f"{name} 第一通道視覺化")
    plt.colorbar()
    plt.show()

# 分析整體統計特性與視覺化
datasets = {
    "Residual Data": residual_data,
    "MV Data": mv_data,
    "Z Data": z_data
}

for name, data in datasets.items():
    analyze_data(data, name)
    analyze_channels(data, name)
    visualize_first_channel(data, name)
