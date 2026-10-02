from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\projects\drone_detection\yolo11n-p2.yaml")

    results = model.train(
        data="VisDrone.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        device=0,
        project=r"D:\projects\drone_detection\runs",
        name="p2_verify",
        save=False,
        plots=False,
    )