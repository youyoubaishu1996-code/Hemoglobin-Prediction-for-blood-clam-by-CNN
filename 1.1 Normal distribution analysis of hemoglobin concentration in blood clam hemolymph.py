# -*- coding: utf-8 -*-
# !/opt/Python-3.10.6
# 当前系统日期：2024/11/14
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 假设 results 是之前生成的结果DataFrame
# 提取实际的血红蛋白浓度值
data = pd.read_csv('./train/血红蛋白浓度值.csv')
actual_hemoglobin = data[['Hemoglobin']].drop_duplicates().values
# 计算均值和标准差
mean = np.mean(actual_hemoglobin)
std_dev = np.std(actual_hemoglobin)

# 打印均值和标准差
print(f"Mean: {mean:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")

# 绘制直方图
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(actual_hemoglobin, bins=20, density=True, alpha=0.6, color='b', edgecolor='black')

# 绘制正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std_dev)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mean = %.2f,  std = %.2f" % (mean, std_dev)
plt.title(title)
plt.xlabel('Hemoglobin Concentration')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# 检查数据是否符合正态分布
# 使用Shapiro-Wilk检验
from scipy.stats import shapiro

stat, p = shapiro(actual_hemoglobin)
print(f'Shapiro-Wilk Test Statistic: {stat:.4f}')
print(f'p-value: {p:.4f}')

# 解释结果
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')