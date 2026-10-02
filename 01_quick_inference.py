from ultralytics import YOLO

# 加载官方预训练模型（第一次运行自动下载 yolo11n.pt，约 5MB）
model = YOLO("yolo11n.pt")

# 用官方示例图做推理（第一次运行自动下载图片）
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,        # 保存画框后的图到 runs/detect/predict/
    conf=0.25,        # 置信度阈值
)

# 看结果里有什么
result = results[0]
print("检测到目标数量:", len(result.boxes))
print("每个目标的类别ID:", result.boxes.cls.tolist())
print("每个目标的置信度:", result.boxes.conf.tolist())
print("每个目标的框坐标(xyxy):", result.boxes.xyxy.tolist())
print("类别名称对照:", result.names)