import cv2
import os
import shutil
import random
import numpy as np

# 视频提取图像
def video_to_frames(video_path, frames_path, sample=1):
    # 打开视频
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    # 创建输出帧目录
    os.makedirs(frames_path, exist_ok=True)
    # 总帧数
    sum_frames=0
    # 采样帧数
    sample_frames=0
    # 采样帧
    while True:
        # 读取一帧
        ok, frame=cap.read()
        if not ok:
            break
        # 采样
        if sum_frames % sample==0:
            save_path=os.path.join(frames_path, f"{sample_frames}.jpg")
            cv2.imwrite(save_path, frame)
            sample_frames+=1
            print(f"Generated {sample_frames} frames...")
        sum_frames+=1
    # 释放视频
    cap.release()
    print(f"Done. Sample {sample_frames} frames into {frames_path}")

# 图像大小调整
def image_resize(image, size=(640, 640), color=(114, 114, 114), cut_fill=False, blur_fill=True):
    # 获取图像信息
    h, w = image.shape[:2]
    target_w, target_h = size
    # 剪裁填充
    if cut_fill:
        # 计算缩放比例
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 从中心裁剪
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        cropped = resized[start_y:start_y+target_h, start_x:start_x+target_w]
        return cropped
    else:
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 缩放
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 模糊填充效果
        if blur_fill:
            # 1. 创建模糊背景底板
            # 将原图缩小到1/8进行模糊处理（进一步降低计算量）
            small_w, small_h = max(1, w // 8), max(1, h // 8)
            
            # 缩小图像
            small_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            
            # 对缩小后的图像进行高斯模糊（增大模糊比例）
            kernel_size = max(3, min(small_w, small_h) // 2)  # 自适应核大小
            if kernel_size % 2 == 0:  # 确保核大小为奇数
                kernel_size += 1
            blurred_small = cv2.GaussianBlur(small_image, (kernel_size, kernel_size), 0)
            
            # 将模糊后的图像拉伸到目标分辨率
            blurred_background = cv2.resize(blurred_small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 2. 将原图按比例缩小放在中间
            # 计算缩放比例
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 缩放原图
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 计算放置位置
            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            
            # 3. 将原图叠加到模糊背景上
            result = blurred_background.copy()
            result[top:top+new_h, left:left+new_w] = resized_image
            
            return result
        else:
            # 创建空背景
            canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
            # 将缩放后的图像放到中心
            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            canvas[top:top+new_h, left:left+new_w] = resized_image
            return canvas

# 批量调整文件夹内所有图像
def batch_resize_folder(input_folder, output_folder, size=(640, 640), cut_fill=False, blur_fill=True):
    # 创建输出目录   
    os.makedirs(output_folder, exist_ok=True)
    # 记录所有图像
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
    # 遍历所有图像
    count = 0
    for fname in image_files:
        # 读取图像
        img_path = os.path.join(input_folder, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue
        # 处理图像
        resized_image = image_resize(image, size, cut_fill=cut_fill, blur_fill=blur_fill)
        save_path = os.path.join(output_folder, fname)
        cv2.imwrite(save_path, resized_image)
        count += 1
        print(f"Processed {count} images...")
    print(f"Done. Total {count} images resized and saved to {output_folder}")

# 图像增强
def augment_image(image, augment_prob=1):
    # 是否进入增强
    if random.random() > augment_prob:
        return image
    # 记录初始信息
    aug_image = image.copy()
    h, w = aug_image.shape[:2]
    # 1️⃣ 随机水平翻转
    if random.random() < 0.4:
        aug_image = cv2.flip(aug_image, 1)
    # 2️⃣ 随机旋转 ±15°
    if random.random() < 0.2:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug_image = cv2.warpAffine(aug_image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # 3️⃣ 随机亮度/对比度
    if random.random() < 0.2:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-30, 30)
        aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
    # 4️⃣ 高斯模糊
    if random.random() < 0.1:
        k = random.choice([3,5])
        aug_image = cv2.GaussianBlur(aug_image, (k,k), 0)
    # 5️⃣ 灰度化
    if random.random() < 0.1:
        gray = cv2.cvtColor(aug_image, cv2.COLOR_BGR2GRAY)
        aug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # 6️⃣ 轻微噪声
    if random.random() < 0.1:
        noise = np.random.normal(0, 0.2, aug_image.shape).astype(np.uint8)
        aug_image = cv2.add(aug_image, noise)
    # CLAHE亮度均衡
    if random.random() < 1.0:
        # 1️⃣ 转换到 LAB 色彩空间
        lab = cv2.cvtColor(aug_image, cv2.COLOR_BGR2LAB)
        # 2️⃣ 分离通道
        l, a, b = cv2.split(lab)
        # 3️⃣ 创建 CLAHE 对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # 4️⃣ 对亮度通道 L 应用 CLAHE
        l = clahe.apply(l)
        # 5️⃣ 合并处理后的通道
        lab = cv2.merge((l, a, b))
        # 6️⃣ 转回 BGR 色彩空间
        aug_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return aug_image

# 批量增强文件夹内所有图像
def batch_augment_folder(input_folder, output_folder, augment_prob=1, repeat_time=1):
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    # 记录所有图像
    image_files = [f for f in os.listdir(input_folder)if f.lower().endswith(('.jpg','.png'))]
    # 遍历所有图像
    count=0
    for fname in image_files:
        # 读取图像
        img_path = os.path.join(input_folder, fname)
        image = cv2.imread(img_path)
        # 处理图像
        for i in range(repeat_time):
            aug_image = augment_image(image, augment_prob=augment_prob)
            base_name, ext = os.path.splitext(fname)
            save_name = f"{base_name}_aug{i+1}{ext}"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, aug_image)
            count += 1
            print(f"Generated {count} augmented images...")
    print(f"Done. Total {count} augmented images saved to {output_folder}")

# 分隔图像
def split_images(input_folder, output_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,seed=0):
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

# 一条龙处理
def dpt_aio(input_video, output_folder, augment=True, split=False, sample=1, blur_fill=True):
    # 创建临时目录
    temp_path=f"{output_folder}/.temp"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path, exist_ok=False)
    # 视频提取图像
    video_to_frames(input_video, temp_path, sample=sample)
    # 批量调整文件夹内所有图像
    batch_resize_folder(temp_path, temp_path, cut_fill=not blur_fill, blur_fill=blur_fill)
    # 批量增强文件夹内所有图像
    if(augment):
        batch_augment_folder(temp_path, output_folder)
    # 删除临时目录
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    # 分隔图像
    if(split):
        split_images(output_folder, output_folder)

if __name__ == "__main__":
    print("This is not an executable file, please do not execute it directly")
