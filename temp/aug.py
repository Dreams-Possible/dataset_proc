import cv2
import os
import random

# 图像处理
def preprocess_image(img, apply_clahe=True, adjust_brightness=True):
    # CLAHE亮度均衡
    if apply_clahe:
        # 1️⃣ 转换到 LAB 色彩空间
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 2️⃣ 分离通道
        l, a, b = cv2.split(lab)
        # 3️⃣ 创建 CLAHE 对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # 4️⃣ 对亮度通道 L 应用 CLAHE
        l = clahe.apply(l)
        # 5️⃣ 合并处理后的通道
        lab = cv2.merge((l, a, b))
        # 6️⃣ 转回 BGR 色彩空间
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 亮度/对比度调整
    if adjust_brightness:
        alpha = 1.1  # 对比度 (1.0-3.0)
        beta = 10    # 亮度 (-100 ~ 100)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# 图像增强
def augment_image(image, augment_prob=1):
    # 是否进入增强
    if random.random() > augment_prob:
        return image
    # 记录初始信息
    aug_image = image.copy()
    h, w = aug_image.shape[:2]
    # 1️⃣ 随机水平翻转
    if random.random() < 0.5:
        aug_image = cv2.flip(aug_image, 1)
    # 2️⃣ 随机旋转 ±15°
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug_image = cv2.warpAffine(aug_image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # 3️⃣ 随机亮度/对比度
    if random.random() < 0.5:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-30, 30)
        aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
    # 4️⃣ 高斯模糊
    if random.random() < 0.2:
        k = random.choice([3,5])
        aug_image = cv2.GaussianBlur(aug_image, (k,k), 0)
    # 5️⃣ 灰度化
    if random.random() < 0.1:
        gray = cv2.cvtColor(aug_image, cv2.COLOR_BGR2GRAY)
        aug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # 6️⃣ 轻微噪声
    if random.random() < 0.1:
        noise = np.random.normal(0, 5, aug_image.shape).astype(np.uint8)
        aug_image = cv2.add(aug_image, noise)
    # 其他处理
    if random.random() < 1.0:
        aug_image = preprocess_image(aug_image)
    return aug_image

# 批量增强文件夹内所有图像
def batch_augment_folder(input_folder, output_folder, augment_prob=1, repeat=1):
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
        for i in range(repeat):
            aug_image = augment_image(image, augment_prob=augment_prob)
            base_name, ext = os.path.splitext(fname)
            save_name = f"{base_name}_aug{i+1}{ext}"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, aug_image)
            count += 1
            print(f"Generated {count} augmented images...")
    print(f"Done. Total {count} augmented images saved to {output_folder}")
