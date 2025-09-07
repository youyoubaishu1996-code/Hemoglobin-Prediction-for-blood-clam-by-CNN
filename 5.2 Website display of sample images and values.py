# -*-coding: Utf-8 -*-
import os
from flask import Flask, render_template
import pandas as pd
import os
from flask import Flask, render_template, request, send_from_directory
app = Flask(__name__)
# 图片存储的目录
# 请确保将 'IMAGE_FOLDER' 替换为你的实际图片目录路径
# IMAGE_FOLDER = dir_message.picture_processed_cropped_output_dir
IMAGE_FOLDER = './pred/test/processed_images'


# 定义你的图片文件夹路径
# 请确保将这些路径替换为你的实际图片目录
IMAGE_DIRS = {
    "input_folder":  './pred/test/origin_images',
    "cropped_folder":  './pred/test/red_cropped_images',
    "processed_cropped_folder":  './pred/test/processed_images'
}


DATA_FILE_PATH= 'pred/test/血红蛋白浓度值.csv'
# 在应用启动时加载数据
# 假设数据文件只有两列，没有header，以制表符分隔
try:
    # 尝试读取数据，并将其存储在app.config中，以便在请求之间共享
    app.config['DATA'] = pd.read_csv(DATA_FILE_PATH, sep='\t', header=None)
    print(f"成功加载数据文件: {DATA_FILE_PATH}")
except FileNotFoundError:
    app.config['DATA'] = pd.DataFrame() # 如果文件不存在，则创建一个空的DataFrame
    print(f"错误：数据文件未找到: {DATA_FILE_PATH}")
except Exception as e:
    app.config['DATA'] = pd.DataFrame()
    print(f"加载数据文件时发生错误: {e}")

@app.route('/')
def index():
    selected_folder_key = request.args.get('folder', 'processed_cropped_folder')

    if selected_folder_key not in IMAGE_DIRS:
        selected_folder_key = 'processed_cropped_folder'

    current_image_folder = IMAGE_DIRS[selected_folder_key]

    image_files = []
    if os.path.exists(current_image_folder):
        image_files = sorted([f for f in os.listdir(current_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

    # 获取数据，并将其转换为字典列表，方便在Jinja2中迭代
    # 假设第一列是图片名，第二列是血红蛋白浓度
    # 注意：这里我们假设图片名在数据的第一列，与图片文件名一致
    data_records = app.config['DATA'].to_dict(orient='records')

    return render_template('图片展示.html',
                           image_files=image_files,
                           selected_folder=selected_folder_key,
                           image_folders=IMAGE_DIRS.keys(),
                           data_records=data_records) # 将数据传递给模板

@app.route('/images/<folder_key>/<filename>')
def serve_image(folder_key, filename):
    if folder_key not in IMAGE_DIRS:
        return "Invalid folder key", 404

    folder_path = IMAGE_DIRS[folder_key]
    return send_from_directory(folder_path, filename)

if __name__ == '__main__':
    # 添加 server_name 参数绕过主机名解析
    app.run(debug=True)

