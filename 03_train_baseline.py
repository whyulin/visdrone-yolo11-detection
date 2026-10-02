from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="VisDrone.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        project=r"D:\projects\drone_detection\runs\trains",
        name="baseline_640_b4",
        patience=10,
        save=True,
        plots=True,
    )