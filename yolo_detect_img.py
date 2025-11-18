# 方法相关
import time
from ultralytics import YOLO
# 类型相关
from dataclasses import dataclass
from typing import List

# 检测数据类
@dataclass
class DetectionData:
    id: int# 编号
    name: str# 名称
    conf: float# 置信度
    x_center: float# x中心点
    y_center: float# y中心点
    width: float# 宽度
    height: float# x高度

# 检测结果类
@dataclass
class DetectionResult:
    time: float
    objects: List[DetectionData]

# yolo检测类
class yolo_detect:

    # 初始化
    def __init__(self, model_path="yolo11n.pt"):
        # 加载模型
        self.model = YOLO(model_path)

    # 运行
    def run(self, image):
        # 开始检测
        start_time = time.time()
        results = self.model(image)
        # 提取检测信息
        boxes = results[0].boxes
        names_dict = results[0].names
        if boxes.cls is not None:
            cls_idx = boxes.cls.cpu().numpy()
        if boxes.conf is not None:
            conf = boxes.conf.cpu().numpy()
        if boxes.xywh is not None:
            xywh = boxes.xywh.cpu().numpy()
        # 打包检测信息
        detected_objects = []
        cost_time = time.time() - start_time
        for id, cf, xy in zip(cls_idx, conf, xywh):
            x_center, y_center, w, h = xy
            cls_name = names_dict[int(id)]
            detected_objects.append(DetectionData(
                int(id),
                str(cls_name),
                float(cf),
                float(x_center),
                float(y_center),
                float(w),
                float(h),
            ))
        # 打包结果信息
        result = DetectionResult(
            time = cost_time,
            objects = detected_objects,
        )
        return result
