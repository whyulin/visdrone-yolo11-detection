from ultralytics.data.utils import check_det_dataset

# 这会触发自动下载（如果第一次跑）并返回数据集配置
data = check_det_dataset("VisDrone.yaml")

print("数据集根目录:", data["path"])
print("训练集路径:", data["train"])
print("验证集路径:", data["val"])
print("类别数:", len(data["names"]))
print("类别名:", data["names"])