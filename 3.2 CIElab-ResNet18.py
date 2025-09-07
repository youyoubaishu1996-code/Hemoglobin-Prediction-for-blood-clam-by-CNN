#file:D:\Project\20250713泥蚶血红蛋白浓度预测模型\CIElab-ResNet8.py
# 修改后的代码，使用自定义RGB到LAB转换，并添加数据筛选功能

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
import time
import warnings

warnings.filterwarnings("ignore")


# Configuration and Seed Setting
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)

# 自定义RGB到LAB转换函数
def rgb_to_lab(rgb_tensor):
    """
    将RGB图像张量转换为LAB色彩空间
    输入: RGB tensor [0, 1] 范围, 形状 (C, H, W)
    输出: LAB tensor, 形状 (C, H, W)
    """
    # 确保输入在[0,1]范围内
    rgb = torch.clamp(rgb_tensor, 0, 1)

    # 转换RGB到XYZ
    # sRGB到XYZ的转换矩阵 (D65光源)
    rgb_to_xyz = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], dtype=torch.float32)

    # 重新排列张量以进行矩阵运算 (C, H, W) -> (H, W, C)
    rgb_permuted = rgb.permute(1, 2, 0)
    shape = rgb_permuted.shape

    # 应用sRGB到线性RGB的转换
    rgb_linear = torch.where(rgb_permuted <= 0.04045,
                            rgb_permuted / 12.92,
                            torch.pow((rgb_permuted + 0.055) / 1.055, 2.4))

    # 转换到XYZ空间
    xyz = torch.matmul(rgb_linear.reshape(-1, 3), rgb_to_xyz.T)
    xyz = xyz.reshape(shape)

    # 归一化XYZ值 (相对于D65白点)
    xyz_norm = xyz / torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)

    # XYZ到LAB转换
    epsilon = 0.008856  # (6/29)^3
    kappa = 903.3      # (29/3)^3

    # 计算f(t)
    f_xyz = torch.where(xyz_norm > epsilon,
                       torch.pow(xyz_norm, 1/3),
                       (kappa * xyz_norm + 16) / 116)

    # 计算LAB值
    L = 116 * f_xyz[:, :, 1] - 16
    a = 500 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
    b = 200 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])

    # 堆叠为 (3, H, W) 格式
    lab = torch.stack([L, a, b], dim=0)

    return lab

# 自定义转换类，将RGB图像转换为CIELAB色彩空间
class RGBtoLAB:
    def __call__(self, img):
        # 将PIL图像或tensor转换为tensor
        if isinstance(img, Image.Image):
            # 转换为tensor并归一化到[0,1]
            img_tensor = transforms.ToTensor()(img)
        else:
            img_tensor = img

        # 转换RGB到LAB
        img_lab = rgb_to_lab(img_tensor)
        return img_lab

# Define image preprocessing pipeline with CIELAB conversion
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda img: img.convert('RGB')),  # 确保图像是RGB格式
    RGBtoLAB(),  # 转换为CIELAB
    transforms.Normalize(mean=[50.0, 0.0, 0.0], std=[50.0, 127.0, 127.0])  # LAB标准化
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Dataset Definition
class HemoglobinImageDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # 读取时仍需要RGB格式

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ResNet18 Model
class ResNetHemoglobinPredictor(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNetHemoglobinPredictor, self).__init__()
        if pretrained:
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18(weights=None)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.to(device)
    train_losses = []
    val_losses = []
    start_time = time.time()

    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training finished. Duration: {training_duration:.2f} seconds")

    return train_losses, val_losses, training_duration


# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predictions.extend(outputs.cpu().numpy().ravel())
            all_labels.extend(labels.cpu().numpy().ravel())

    avg_test_loss = test_loss / len(test_loader)
    predictions = np.array(predictions)
    all_labels = np.array(all_labels)

    mse = mean_squared_error(all_labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, predictions)
    r2 = r2_score(all_labels, predictions)

    print('\n--- Evaluation Metrics ---')
    print(f'Test Loss (MSE): {avg_test_loss:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R-squared (R2): {r2:.4f}')

    return predictions, all_labels, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'test_loss': avg_test_loss}


# Visualization Functions
def plot_loss_curves(train_losses, val_losses):
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['svg.fonttype'] = 'none'  # 确保文本在SVG中可编辑

    plt.figure(figsize=(5, 5), dpi=900)  # 提高DPI确保清晰度

    # 绘制训练损失曲线（蓝色实线，加粗）
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             'b-', linewidth=2.5, label='Train Loss')

    # 绘制验证损失曲线（橙色实线，加粗）
    plt.plot(range(1, len(val_losses) + 1), val_losses,
             color='orange', linestyle='-', linewidth=2.5, label='Validation Loss')

    plt.title('Training and Validation Loss Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10, fontweight='bold')
    plt.ylabel('Loss (MSE)', fontsize=10, fontweight='bold')

    # 设置刻度标签为粗体
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置图例（新罗马字体，加粗）
    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 10})

    # 保存为矢量图
    plt.savefig('loss_curves.svg', format='svg', bbox_inches='tight')
    print("Saved vector image: loss_curves.svg")

    plt.show()

def plot_predictions(predictions, actual_labels, metrics):
    # 设置全局字体为Times New Roman并加粗
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['svg.fonttype'] = 'none'  # 确保文本在SVG中可编辑

    plt.figure(figsize=(8, 6), dpi=900)  # 提高DPI确保清晰度

    # 绘制散点图（点大小增加，透明度降低）
    sns.scatterplot(x=actual_labels, y=predictions, alpha=0.7, s=60)

    # 设置坐标轴标签（加粗）
    plt.xlabel('Actual Hemoglobin Levels', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Hemoglobin Levels', fontsize=12, fontweight='bold')

    # 设置标题（加粗）
    plt.title(f'Actual vs Predicted\n(R²: {metrics["r2"]:.2f}, RMSE: {metrics["rmse"]:.2f})',
              fontsize=14, fontweight='bold')

    # 计算最小值和最大值
    min_val = min(min(actual_labels), min(predictions))
    max_val = max(max(actual_labels), max(predictions))

    # 绘制理想参考线（加粗红色虚线）
    plt.plot([min_val, max_val], [min_val, max_val],
             color='red', linestyle='--', linewidth=2.5,
             label='Perfect Prediction')

    # 设置网格（半透明虚线）
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置刻度标签为粗体
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    # 设置图例（Times New Roman字体，加粗）
    plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 11})

    # 保存为矢量图
    plt.savefig('predictions.svg', format='svg', bbox_inches='tight')
    print("Saved vector image: predictions.svg")

    plt.show()


# Main Execution
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('./train/血红蛋白浓度值.csv')

    # 数据筛选：只选择血红蛋白值在10到30之间的数据
    print(f"原始数据量: {len(data)}")
    filtered_data = data[(data.iloc[:, 1] >= 10) & (data.iloc[:, 1] <= 30)]
    print(f"筛选后数据量 (血红蛋白值在10-30之间): {len(filtered_data)}")
    print(f"筛选掉的数据量: {len(data) - len(filtered_data)}")

    # 显示筛选前后的统计信息
    print("\n原始数据血红蛋白浓度统计:")
    print(f"  最小值: {data.iloc[:, 1].min():.2f}")
    print(f"  最大值: {data.iloc[:, 1].max():.2f}")
    print(f"  平均值: {data.iloc[:, 1].mean():.2f}")

    print("\n筛选后数据血红蛋白浓度统计:")
    print(f"  最小值: {filtered_data.iloc[:, 1].min():.2f}")
    print(f"  最大值: {filtered_data.iloc[:, 1].max():.2f}")
    print(f"  平均值: {filtered_data.iloc[:, 1].mean():.2f}")

    # Create dataset
    full_dataset = HemoglobinImageDataset(
        data=filtered_data,  # 使用筛选后的数据
        image_dir='./train/processed_images',
        transform=transform
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ResNetHemoglobinPredictor(num_classes=1, pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train model
    num_epochs = 50
    train_losses, val_losses, training_duration = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )
    # Create model directory if it doesn't exist
    os.makedirs('model/ResNet18', exist_ok=True)
    # Save the trained model
    model_save_path = 'model/ResNet18/best_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_duration': training_duration
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    # Evaluate model
    predictions, actual_labels, metrics = evaluate_model(model, test_loader)

    # Visualize results
    plot_loss_curves(train_losses, val_losses)
    plot_predictions(predictions, actual_labels, metrics)
