import os
import random
import shutil

# 分隔图像
def split_images(input_folder:str, output_folder:str, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,seed=0):
    # 比例检测
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
    # 创建输出目录
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_folder, subset), exist_ok=True)
    # 读取所有图片
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg",".png"))]
    # 乱序（可复现）
    random.seed(seed)
    random.shuffle(image_files)
    # 划分比例
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_files = image_files[:train_end]
    val_files   = image_files[train_end:val_end]
    test_files  = image_files[val_end:]
    # 复制函数
    def copy_to(files, subset_name):
        for name in files:
            src = os.path.join(input_folder, name)
            dst = os.path.join(output_folder, subset_name, name)
            shutil.copy(src, dst)
    # 复制
    copy_to(train_files, "train")
    copy_to(val_files,   "val")
    copy_to(test_files,  "test")
    # 输出结果
    print("Dataset split finished")
    print(f"Total images: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")
    print("Saved to:", output_folder)
