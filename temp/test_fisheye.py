import cv2
import numpy as np
from dtst_proc_tool import augment_image

# 创建一个测试图像
def create_test_image():
    # 创建一个彩色网格图像，便于观察畸变效果
    height, width = 400, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建网格效果
    grid_size = 40
    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            color = [
                int(255 * i / height),      # 红色渐变
                int(255 * j / width),       # 绿色渐变  
                128                         # 固定蓝色
            ]
            cv2.rectangle(image, (j, i), (j+grid_size, i+grid_size), color, -1)
            cv2.rectangle(image, (j, i), (j+grid_size, i+grid_size), (255, 255, 255), 1)
    
    return image

def main():
    # 创建测试图像
    test_image = create_test_image()
    
    print("测试鱼眼畸变效果...")
    
    # 保存原始图像
    cv2.imwrite("test_original.jpg", test_image)
    
    # 多次测试鱼眼畸变效果
    for i in range(5):
        # 强制启用鱼眼畸变（设置高概率）
        aug_image = augment_image(test_image, augment_prob=1.0)
        cv2.imwrite(f"test_fisheye_{i+1}.jpg", aug_image)
        print(f"生成第 {i+1} 个鱼眼畸变测试图像")
    
    print("测试完成！")
    print("生成的文件：")
    print("- test_original.jpg: 原始网格图像")
    print("- test_fisheye_*.jpg: 鱼眼畸变效果图像")

if __name__ == "__main__":
    main()
