import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# 导入你的模型定义，确保这些文件在同一目录下或PYTHONPATH中
from SimpleCNN import SimpleCNNPredictor
from UNet import UNetHemoglobinPredictor
from ResNet18 import ResNetHemoglobinPredictor
from YOLOLike import YOLOLikeHemoglobinPredictor

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于flash消息的秘钥，生产环境请替换为更安全的
app.config['UPLOAD_FOLDER'] = './static/uploads'  # 临时上传目录
app.config['PROCESSED_FOLDER'] = 'static/processed'  # 预处理后的图片保存目录
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传和处理目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('model/SimpleCNN', exist_ok=True)
os.makedirs('model/UNet', exist_ok=True)
os.makedirs('model/ResNet18', exist_ok=True)
os.makedirs('model/YOLOLike', exist_ok=True)

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 图像预处理函数 (从你之前提供的代码复制) ---
def adaptive_histogram_equalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def crop_red_area(image):
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_red = image[y:y + h, x:x + w]
        return cropped_red
    else:
        return None


def process_image(image):
    image_eq = adaptive_histogram_equalization(image)
    image_wb = white_balance(image_eq)
    image_normalized = normalize_image(image_wb)
    processed_image = (image_normalized * 255).astype(np.uint8)
    return processed_image


def preprocess_for_prediction(image_path, output_dir):
    """
    对单个图像进行裁剪红色区域并进行预处理。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    cropped_red = crop_red_area(image)
    if cropped_red is not None:
        processed_img_data = process_image(cropped_red)
        # 将OpenCV格式的图像转换为PIL Image，以便后续的ToTensor等操作
        processed_pil_img = Image.fromarray(cv2.cvtColor(processed_img_data, cv2.COLOR_BGR2RGB))

        # 保存预处理后的图像，以便调试或查看
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_img_data)

        return processed_pil_img
    else:
        return None


# --- 模型加载和预测函数 (从你之前提供的代码复制) ---
# Transform for inference
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.406])
])


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

    if not os.path.exists(model_paths[model_name]):
        flash(f"Error: Model file not found at {model_paths[model_name]}", "error")
        return None

    # Load the appropriate model architecture
    if model_name == 'SimpleCNN':
        model = SimpleCNNPredictor()
    elif model_name == 'UNet':
        model = UNetHemoglobinPredictor(n_channels=3, n_classes=1)
    elif model_name == 'ResNet18':
        model = ResNetHemoglobinPredictor(num_classes=1,
                                          pretrained=False)  # pretrained=True will download ImageNet weights
    elif model_name == 'YOLOLike':
        model = YOLOLikeHemoglobinPredictor()

    try:
        checkpoint = torch.load(model_paths[model_name], map_location=device)
        # 兼容性处理：如果保存的状态字典包含 'model_state_dict' 键，则使用它
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # 否则直接加载
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        flash(f"Error loading model weights for {model_name}: {str(e)}", "error")
        return None


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    original_image_url = None
    processed_image_url = None
    selected_model = None

    if request.method == 'POST':
        # Check if a model was selected
        selected_model = request.form.get('model_select')
        if not selected_model:
            flash('Please select a model.', 'error')
            return redirect(request.url)

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_filepath)
            original_image_url = url_for('uploaded_file', filename=filename)

            # Preprocess the image
            processed_pil_image = preprocess_for_prediction(original_filepath, app.config['PROCESSED_FOLDER'])

            if processed_pil_image is None:
                flash('Could not find red area or process image.', 'error')
                return redirect(request.url)

            # Define processed image URL for display
            processed_filename = os.path.basename(original_filepath)
            processed_image_url = url_for('processed_file', filename=processed_filename)

            # Load model
            model = load_model(selected_model)
            if model is None:
                return redirect(request.url)  # Error message already flashed by load_model

            # Prepare image for model inference
            try:
                img_tensor = transform(processed_pil_image).unsqueeze(0).to(device)  # Add batch dimension

                with torch.no_grad():
                    output = model(img_tensor).squeeze().cpu().numpy()

                prediction_result = f"{output:.2f} g/dL"  # Format prediction
                flash(f"Prediction successful using {selected_model}!", 'success')
            except Exception as e:
                flash(f"Error during prediction: {str(e)}", 'error')
                prediction_result = "Prediction Failed"
        else:
            flash('Allowed image types are png, jpg, jpeg, gif', 'error')

    return render_template('index.html',
                           prediction_result=prediction_result,
                           original_image_url=original_image_url,
                           processed_image_url=processed_image_url,
                           selected_model=selected_model)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed_images/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


from flask import send_from_directory

if __name__ == '__main__':
    app.run(debug=True)