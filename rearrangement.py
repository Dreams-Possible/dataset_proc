import os
import shutil

# 批量调整文件夹内所有图像
def rearrangement_image(input_folder, output_folder):
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    # 记录所有图像
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
    # 遍历所有图像
    count = 0
    for image in image_files:
        # 加载信息
        ext = os.path.splitext(image)[1]
        new_name = f"{count:05d}{ext}"
        # 保存图像
        src = os.path.join(input_folder, image)
        dst = os.path.join(output_folder, new_name)
        shutil.copy(src, dst)
        count += 1
        print(f"Processed {count} images...")
    print(f"Done. Total {count} images rearranged and saved to {output_folder}")
