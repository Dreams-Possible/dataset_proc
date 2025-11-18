
import os
import cv2
import yolo_detect_img_v1 as yolo_detect

# 自动标注
def run(model_path,dataset_path,lable_path):
    # 例化yolo检测器
    yoloer=yolo_detect.yolo_detect(model_path=model_path)
    # 创建输出目录
    os.makedirs(lable_path, exist_ok=True)
    # 记录所有图像
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png'))]
    # 遍历所有图像
    count = 0
    for image in image_files:
        # 读取图像
        img_path = os.path.join(dataset_path, image)
        # 检测
        result=yoloer.run(image=img_path)
        # 基本信息
        print(f"cost: {result.time:.2f} seconds")
        print(f"detected {len(result.objects)} targets:")
        # 检测结果
        for obj in result.objects:
            print(f"class: {obj.name} conf: {obj.conf:.2f} pos: ({obj.x_center:.1f}, {obj.y_center:.1f}) width and height: ({obj.width:.1f}, {obj.height:.1f})")
        # 记录lable
        label_name = os.path.splitext(image)[0] + ".txt"
        label_path = os.path.join(lable_path, label_name)
        # 读取图像尺寸用于归一化
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(label_path, "w", encoding="utf-8") as f:
            for obj in result.objects:
                # 记录ID
                class_id = obj.id
                # 归一化尺寸
                x = obj.x_center / w
                y = obj.y_center / h
                ww = obj.width / w
                hh = obj.height / h
                # 写入文件
                f.write(f"{class_id} {x:.6f} {y:.6f} {ww:.6f} {hh:.6f}\n")
        count += 1
        print(f"Processed {count} images...")
    print(f"Done. Total {count} images labled and saved to {lable_path}")


