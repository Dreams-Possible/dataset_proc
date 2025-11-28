import cv2
import numpy as np
from dtst_proc_tool import image_resize

# 创建一个测试图像
def create_test_image():
    # 创建一个彩色渐变图像
    height, width = 400, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建渐变效果
    for i in range(height):
        for j in range(width):
            image[i, j] = [
                int(255 * i / height),      # 红色渐变
                int(255 * j / width),       # 绿色渐变  
                int(255 * (i+j) / (height+width))  # 蓝色渐变
            ]
    
    return image

def main():
    # 创建测试图像
    test_image = create_test_image()
    
    # 测试不同的填充方式
    print("测试不同的图像填充方式...")
    
    # 1. 默认填充（YOLO灰色）
    default_fill = image_resize(test_image, size=(640, 640), cut_fill=False, blur_fill=False)
    
    # 2. 模糊填充
    blur_fill = image_resize(test_image, size=(640, 640), cut_fill=False, blur_fill=True)
    
    # 3. 裁剪填充
    crop_fill = image_resize(test_image, size=(640, 640), cut_fill=True, blur_fill=False)
    
    # 保存结果
    cv2.imwrite("test_default_fill.jpg", default_fill)
    cv2.imwrite("test_blur_fill.jpg", blur_fill)
    cv2.imwrite("test_crop_fill.jpg", crop_fill)
    
    print("测试完成！")
    print("生成的文件：")
    print("- test_default_fill.jpg: 默认灰色填充")
    print("- test_blur_fill.jpg: 模糊填充效果")
    print("- test_crop_fill.jpg: 裁剪填充")
    
    # 显示图像信息
    print(f"\n原始图像尺寸: {test_image.shape}")
    print(f"默认填充尺寸: {default_fill.shape}")
    print(f"模糊填充尺寸: {blur_fill.shape}")
    print(f"裁剪填充尺寸: {crop_fill.shape}")

if __name__ == "__main__":
    main()
