from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n-p2.yaml")   # 不加载预训练

    results = model.train(
        data="VisDrone.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        project=r"D:\projects\drone_detection\runs\trains",
        name="yolo11n_p2_640",          # 对应你之前的从零实验
        patience=10,
        save=True,
        plots=True,
    )