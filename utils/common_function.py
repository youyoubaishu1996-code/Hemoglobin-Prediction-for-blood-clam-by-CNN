import cv2
import numpy as np
import os
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
    """
    从给定的图片中裁剪出红色区域。

    参数:
    - image: 输入图像数据

    返回:
    - 裁剪后的红色区域图像数据，如果没有找到红色区域则返回 None
    """
    # 定义红色在HSV颜色空间中的范围
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
        print("No red area found.")
        return None


def process_image(image):
    """
    对图像进行预处理（直方图均衡化、白平衡、归一化）。

    参数:
    - image: 输入图像数据

    返回:
    - 处理后的图像数据
    """
    # 直方图均衡化
    image_eq = adaptive_histogram_equalization(image)

    # 白平衡
    image_wb = white_balance(image_eq)

    # 归一化
    image_normalized = normalize_image(image_wb)

    # 将归一化后的图像值转换回 [0, 255] 范围内，并转换为 uint8 类型
    processed_image = (image_normalized * 255).astype(np.uint8)
    return processed_image


def preprocess_images(input_dir, output_dir, process_func):
    """
    遍历指定目录下的所有图像文件，并对每个图像文件调用指定的处理函数。

    参数:
    - input_dir: 输入图像文件的目录
    - output_dir: 处理后图像文件的输出目录
    - process_func: 用于处理单个图像文件的函数
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to read image: {input_path}")
                continue

            processed_image = process_func(image)
            if processed_image is not None:
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved: {output_path}{filename}")
            else:
                print(f"No red area found in: {filename}")


