import yolo_ai2ai_v1 as ya2a

# 模型目录
model_path="./last.pt"
# 数据集目录
dataset_path="./dataset"
# 标签目录
lable_path="./lable"

# 自动标注
ya2a.run(
    model_path=model_path,
    dataset_path=dataset_path,
    lable_path=lable_path)
