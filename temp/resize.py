import cv2
import os
import numpy as np

# 图像大小调整
def image_resize(image, size=(640, 640), color=(0, 0, 0), fill=False):
    # 获取图像信息
    h, w = image.shape[:2]
    target_w, target_h = size
    # 填充剪裁
    if fill:
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
        # 创建空白背景
        canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
        # 将缩放后的图像放到中心
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized_image
        return canvas

# 批量调整文件夹内所有图像
def batch_resize_folder(input_folder, output_folder, size=(640, 640), fill=False):
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
        resized_image = image_resize(image, size, fill=fill)
        save_path = os.path.join(output_folder, fname)
        cv2.imwrite(save_path, resized_image)
        count += 1
        print(f"Processed {count} images...")
    print(f"Done. Total {count} images resized and saved to {output_folder}")
