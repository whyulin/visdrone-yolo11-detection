import urllib.request
import os

url = "https://huggingface.co/davsolai/yolo11x-p2-coco/raw/main/yolo11-p2.yaml"
save_path = "yolo11n-p2.yaml"

if not os.path.exists(save_path):
    print("开始下载...")
    urllib.request.urlretrieve(url, save_path)
    print(f"下载完成: {save_path}")
else:
    print(f"文件已存在: {save_path}")