# -*- coding: utf-8 -*-
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import re

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_image_dir = 'pred/test/processed_images'
# Dataset class
class HemoglobinImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # 调试信息
        print(f"\nInitializing Dataset with directory: {image_dir}")
        print(f"Directory exists: {os.path.exists(image_dir)}")

        if os.path.exists(image_dir):
            all_files = os.listdir(image_dir)
            print(f"All files in directory ({len(all_files)}): {all_files}")

            self.image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"Image files found ({len(self.image_files)}): {self.image_files}")
        else:
            print("Directory does not exist!")
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_name
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回一个空白图像或其他处理
            blank_image = torch.zeros(3, 64, 64)  # 示例：返回黑色图像
            return blank_image, img_name


# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.406])
])

# Create dataset and dataloader
test_dataset = HemoglobinImageDataset(test_image_dir, transform=transform)
print(len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model loading function
def load_model(model_name):
    model_paths = {
        'SimpleCNN': 'model/SimpleCNN/best_model.pth',
        'UNet': 'model/UNet/best_model.pth',
        'ResNet18': 'model/ResNet18/best_model.pth',
        'YOLOLike': 'model/YOLOLike/best_model.pth'
    }

    if model_name not in model_paths:
        raise ValueError(f"Unknown model name: {model_name}. Available options: {list(model_paths.keys())}")

    # Load the appropriate model architecture
    if model_name == 'SimpleCNN':
        from SimpleCNN import SimpleCNNPredictor
        model = SimpleCNNPredictor()
    elif model_name == 'UNet':
        from UNet import UNetHemoglobinPredictor
        model = UNetHemoglobinPredictor(n_channels=3, n_classes=1)
    elif model_name == 'ResNet18':
        from ResNet18 import ResNetHemoglobinPredictor
        model = ResNetHemoglobinPredictor(num_classes=1, pretrained=True)
    elif model_name == 'YOLOLike':
        from YOLOLike import YOLOLikeHemoglobinPredictor
        model = YOLOLikeHemoglobinPredictor()

    # Load model weights
    checkpoint = torch.load(model_paths[model_name], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


# Main prediction function
def predict_and_save(model_name):
    # Load model
    model = load_model(model_name)
    # Make predictions
    all_preds = []
    all_img_names = []

    with torch.no_grad():
        for inputs, img_names in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()

            # Move predictions to CPU and convert to numpy
            preds = outputs.cpu().numpy()

            # Handle single prediction case
            if preds.ndim == 0:
                all_preds.append(float(preds))
                all_img_names.append(img_names[0])  # Single image case
            else:
                all_preds.extend(preds.tolist())
                all_img_names.extend(img_names)

            # 创建结果DataFrame
        results = pd.DataFrame({
            'Algorithm': model_name,
            'Image': all_img_names,
            'Predicted Hemoglobin Level': all_preds
        })

        # === 修复后的排序逻辑 ===
        # 1. 提取文件名中的数字部分（移除所有非数字字符）
        results['SortKey'] = results['Image'].apply(
            lambda x: int(re.sub(r'\D', '', os.path.splitext(x)[0]))
        )

        # 2. 按数字值升序排序
        results = results.sort_values(by='SortKey')

        # 3. 删除临时排序列
        results = results.drop(columns=['SortKey'])
        # ======================

        # 保存结果（保持不变）
        output_csv = f'./pred/test/血红蛋白浓度预测值_{model_name}.csv'
        results.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
        print(results.head())


# Example usage
if __name__ == "__main__":
    model_name = 'SimpleCNN'  # Change this to 'UNet', 'ResNet18', or 'YOLOLike' as needed
    predict_and_save(model_name)