# -*- coding: utf-8 -*-
# !/opt/Python-3.10.6
# 当前系统日期：2024/11/14
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
# ====== 插入点1：导入scipy.stats ======
from scipy.stats import pearsonr
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# ====== 插入结束 ======

# ====== 插入点2：设置全局字体样式 ======
# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'  # 加粗
plt.rcParams['font.size'] = 20  # 基础字体大小增大到20
# ====== 插入结束 ======

# 读取Excel文件
data = pd.read_csv('./train/血红蛋白浓度值.csv')

# 假设CSV文件中的列名是'Image'和'Hemoglobin'
image_col = 'Image'
hemoglobin_col = 'Hemoglobin'
# 指定图像文件夹路径
image_dir = './train/processed_images'

# 创建一个新的DataFrame用于存储颜色特征
color_features = []

# 提取颜色特征（删除色差和标准差指标）
for index, row in data.iterrows():
    img_name = row[0]
    img_path = os.path.join(image_dir, img_name)

    # 加载图像
    image = Image.open(img_path).convert("RGB")
    image_array = np.array(image).astype(np.float32)  # 转换为浮点型以便计算

    # ====== 修改点1：只保留亮度和饱和度指标 ======
    # 1. 亮度 (Brightness) - 计算为RGB三通道的平均值
    brightness = np.mean(image_array) / 255.0  # 归一化到[0,1]

    # 2. 色调饱和度 (Saturation)
    # 转换为HSV颜色空间并提取饱和度通道
    hsv_image = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv_image[:, :, 1]) / 255.0  # 归一化到[0,1]
    # ====== 修改结束 ======

    # 计算每个通道的均值
    red_mean = np.mean(image_array[:, :, 0])
    green_mean = np.mean(image_array[:, :, 1])
    blue_mean = np.mean(image_array[:, :, 2])

    # 计算各通道比率
    total = red_mean + green_mean + blue_mean + 1e-8  # 防止除零错误
    red_ratio = red_mean / total
    green_ratio = green_mean / total
    blue_ratio = blue_mean / total

    # 将颜色特征添加到列表中
    color_features.append({
        'Image': img_name,
        'Red_Mean': red_mean,
        'Green_Mean': green_mean,
        'Blue_Mean': blue_mean,
        'Red_Ratio': red_ratio,
        'Green_Ratio': green_ratio,
        'Blue_Ratio': blue_ratio,
        # ====== 修改点2：只保留亮度和饱和度指标 ======
        'Brightness': brightness,
        'Saturation': saturation,
        # ====== 修改结束 ======
        'Hemoglobin': row[hemoglobin_col]
    })

# 将颜色特征转换为DataFrame
df_color_features = pd.DataFrame(color_features)

# 计算皮尔逊相关系数（删除色差和标准差指标）
# ====== 修改点：在相关系数矩阵中删除RG_diff和Std_RGB ======
correlation_matrix = df_color_features[[
    'Red_Mean', 'Green_Mean', 'Blue_Mean',
    'Red_Ratio', 'Green_Ratio', 'Blue_Ratio',
    'Brightness', 'Saturation',
    'Hemoglobin'
]].corr()

print(correlation_matrix)

# ====== 插入点4：替换原热力图代码 - 开始 ======
# 计算显著性p值矩阵
p_matrix = np.zeros(correlation_matrix.shape)
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        if i != j:  # 跳过对角线
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            r, p_value = pearsonr(df_color_features[col1], df_color_features[col2])
            p_matrix[i, j] = p_value

# 创建显著性星标矩阵
star_matrix = np.full(correlation_matrix.shape, '', dtype=object)
for i in range(p_matrix.shape[0]):
    for j in range(p_matrix.shape[1]):
        if i != j:  # 跳过对角线
            p = p_matrix[i, j]
            if p < 0.001:
                star_matrix[i, j] = '***'
            elif p < 0.01:
                star_matrix[i, j] = '**'
            elif p < 0.05:
                star_matrix[i, j] = '*'
            else:
                star_matrix[i, j] = 'ns'  # 不显著

# 创建组合注释矩阵（左下显示数值，右上显示圆圈）
annot_matrix = np.empty_like(correlation_matrix, dtype=object)
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        if i >= j:  # 左下三角（包括对角线）
            # 格式：数值（保留两位小数）
            value = correlation_matrix.iloc[i, j]
            annot_matrix[i, j] = f"{value:.2f}"  # 仅显示数值，保留两位小数
        else:  # 右上三角
            annot_matrix[i, j] = ''  # 留空，用圆圈表示

# 创建热力图掩码（只显示下三角）
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# 创建图形（保持尺寸以适应特征）
plt.figure(figsize=(24, 24))  # 稍减小尺寸

# 创建全白的颜色映射（保持不变）
white_cmap = ListedColormap(['white'])

# 绘制左下部分（数值表格）（保持不变）
sns.heatmap(
    correlation_matrix,
    mask=mask,  # 使用定义好的mask变量
    annot=annot_matrix,
    fmt='',
    cmap=white_cmap,
    cbar=False,
    annot_kws={
        'fontsize': 32,
        'ha': 'center',
        'va': 'center',
        'fontname': 'Times New Roman',
        'fontweight': 'bold'
    },
    linewidths=1.5,
    linecolor='black'
)

correlation_cmap = LinearSegmentedColormap.from_list(
    'correlation_cmap',
    ['#203b74', '#FFFFFF', '#ac1d34'],
    N=256
)
# ====== 修改结束 ======

# 添加右上部分（圆圈表示相关性）
# 计算圆圈大小和颜色
circle_sizes = (-np.log10(p_matrix + 1e-10) * 800)  # 大小基于显著性（p值的负对数）
circle_colors = correlation_matrix.values  # 颜色基于相关系数

# 绘制圆圈
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        if i < j:  # 右上三角
            x = j + 0.5
            y = i + 0.5
            size = circle_sizes[i, j]
            color_val = circle_colors[i, j]

            # 标准化颜色值到[-1,1]范围
            norm_color = (color_val + 1) / 2  # 映射到[0,1]
            color = correlation_cmap(norm_color)

            # 绘制圆圈
            plt.scatter(
                x, y,
                s=size,
                c=[color],
                alpha=0.8,
                edgecolors='black',
                linewidths=4  # 加粗圆圈边框
            )

            # 添加星标
            stars = star_matrix[i, j]
            plt.text(
                x, y,
                stars,
                ha='center',
                va='center',
                fontsize=32,
                fontweight='bold',
                fontname='Times New Roman',
                color='black'  # 强制使用黑色星标
            )

# ====== 修改点2：调整图例位置更贴近图表 ======
# 添加颜色条图例（减小pad值使其更贴近图表）
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=correlation_cmap,
                                         norm=plt.Normalize(vmin=-1, vmax=1)),
                    ax=plt.gca(),
                    label='Correlation Coefficient',
                    fraction=0.05,
                    pad=0.02)  # 减小pad值使图例更贴近图表

# 设置图例字体
cbar.ax.yaxis.label.set_fontproperties({
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': 36
})
cbar.ax.tick_params(labelsize=32)

# 添加图例说明（位置稍作调整）
plt.text(0.5, -0.12,  # 提高位置
         "Circle size: -log10(p-value)\n"
         "***: p<0.001, **: p<0.01, *: p<0.05",
         ha='center', va='top',
         fontsize=32, fontname='Times New Roman',
         transform=plt.gca().transAxes)

# 设置坐标轴标签（加粗、放大）（保持不变）
tick_labels = correlation_matrix.columns
plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels,
           fontsize=28, fontweight='bold', fontname='Times New Roman')
plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels,
           fontsize=28, fontweight='bold', fontname='Times New Roman')

plt.tight_layout()
plt.savefig('./correlation_matrix_enhanced.png', dpi=600, bbox_inches='tight')
plt.show()