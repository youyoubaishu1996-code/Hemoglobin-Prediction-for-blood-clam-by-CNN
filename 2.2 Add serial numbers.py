import cv2
import numpy as np
import pygetwindow as gw
import os
def load_and_label_contours(input_path, output_dir='output_samples'):
    # 读取图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"错误：无法读取图像 {input_path}")
        return
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 获取屏幕尺寸
    try:
        screen = gw.getWindowsWithTitle('Program Manager')[0]
        screen_width, screen_height = screen.width, screen.height
    except:
        screen_width, screen_height = 1920, 1080  # 默认尺寸

    # 设置窗口
    cv2.namedWindow('Contour Labeling', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Contour Labeling', 800, 600)
    cv2.moveWindow('Contour Labeling', screen_width - 810, 10)
    cv2.setWindowProperty('Contour Labeling', cv2.WND_PROP_TOPMOST, 1)

    # 颜色检测
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define HSV range for red color
    lower_red1 = np.array([0, 50, 30])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 50, 30])
    upper_red2 = np.array([180, 255, 255])
    # 创建各个颜色的掩膜
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    labeled_image = image.copy()  # 用于保存完整标注图

    # 用于存储已标注的样本信息
    labeled_samples = []
    current_index = 0

    while current_index < len(contours):
        contour = contours[current_index]
        x, y, w, h = cv2.boundingRect(contour)

        # 绘制当前轮廓
        display_image = labeled_image.copy()
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Contour Labeling', display_image)
        cv2.waitKey(300)

        # 用户输入
        user_input = input(f"请输入第{current_index + 1}个轮廓的序号(输入quit退出, 输入back返回上一步): ").strip()

        if user_input.lower() == 'quit':
            print("提前退出...")
            break
        elif user_input.lower() == 'back':
            if current_index > 0:
                # 删除上一步保存的文件
                if labeled_samples:
                    last_label = labeled_samples.pop()
                    last_file = f"{output_dir}/{last_label}.png"
                    if os.path.exists(last_file):
                        os.remove(last_file)
                        print(f"已删除样本: {last_file}")
                current_index -= 1
                # 重新加载图像到上一步状态
                labeled_image = image.copy()
                for idx, label in enumerate(labeled_samples):
                    contour = contours[idx]
                    x, y, w, h = cv2.boundingRect(contour)
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    label_x = int(x + (w - label_size[0]) / 2)
                    label_y = y + h + label_size[1]
                    cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(labeled_image, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                continue
            else:
                print("已经是第一步，无法返回")
                continue

        # 保存单独样本
        sample = image[max(y, 0):min(y + h, image.shape[0]),
                 max(x, 0):min(x + w, image.shape[1])]
        black_mask = np.all(sample == [0, 0, 0], axis=-1)
        sample[black_mask] = [255, 255, 255]
        cv2.imwrite(f"{output_dir}/{user_input}.png", sample)
        print(f"已保存样本: {output_dir}/{user_input}.png")
        labeled_samples.append(user_input)

        # 更新完整标注图
        label_size, _ = cv2.getTextSize(user_input, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        label_x = int(x + (w - label_size[0]) / 2)
        label_y = y + h + label_size[1]

        cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(labeled_image, user_input, (label_x-10, label_y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

        current_index += 1

    # 保存完整标注图
    final_output = os.path.join(output_dir, "final_labeled.png")
    cv2.imwrite(final_output, labeled_image)
    print(f"\n标注完成！结果保存在: {output_dir}/")
    print(f"- 单个样本: 序号.png")
    print(f"- 完整标注: final_labeled.png")

    cv2.imshow('Contour Labeling', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_and_label_contours('contours_output.png', 'samples')