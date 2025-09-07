# -*- coding: utf-8 -*-
# 当前系统日期：2024/12/9
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F  # 导入 F 模块
# 定义图像预处理流程
def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),  # 将图像缩放至64x64像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
    ])

# 定义数据集类
class HemoglobinImageDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        """
        初始化数据集类，设置数据、图像目录路径及图像变换
        :param data: 包含图像名和标签的数据框
        :param image_dir: 图像所在目录
        :param transform: 应用于图像的变换
        """
        self.data = data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引处的数据项
        :param idx: 数据项索引
        :return: 图像及其对应的标签
        """
        img_name = self.data.iloc[idx, 0]  # 假设第一列为图像文件名
        label = self.data.iloc[idx, 1]  # 假设第二列为标签（血红蛋白浓度）

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # 打开并转换图像为RGB模式

        if self.transform:
            image = self.transform(image)  # 如果定义了变换，则对图像应用变换

        return image, torch.tensor(label, dtype=torch.float32)  # 返回图像张量和标签张量
# 定义测试数据集
class TestHemoglobinImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name
# 定义模型
class HemoglobinPredictor(nn.Module):
    def __init__(self):
        super(HemoglobinPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 第一个卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 最大池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 第二个卷积层
        self.fc1 = nn.Linear(32 * 16 * 16, 120)  # 全连接层1
        self.fc2 = nn.Linear(120, 84)  # 全连接层2
        self.fc3 = nn.Linear(84, 1)  # 输出层

    def forward(self, x):
        """
        定义前向传播过程
        :param x: 输入张量
        :return: 输出张量
        """
        x = self.pool(F.relu(self.conv1(x)))  # 卷积->激活->池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积->激活->池化
        x = x.view(-1, 32 * 16 * 16)  # 展平张量
        x = F.relu(self.fc1(x))  # 全连接->激活
        x = F.relu(self.fc2(x))  # 全连接->激活
        x = self.fc3(x)  # 输出层
        return x
# 定义函数以返回所需的组件
def get_model_transform():
    """
    返回图像预处理转换、神经网络模型和数据集类。
    """
    transform = get_transform()
    return transform

def get_model_model():
    """
    返回图像预处理转换、神经网络模型和数据集类。
    """
    model = HemoglobinPredictor()
    return model

def get_model_dataset_class():
    """
    返回图像预处理转换、神经网络模型和数据集类。
    """
    dataset_class = HemoglobinImageDataset
    return dataset_class


def get_test_dataset_class():
    """
    返回图像预处理转换、神经网络模型和数据集类。
    """
    dataset_class = TestHemoglobinImageDataset
    return dataset_class