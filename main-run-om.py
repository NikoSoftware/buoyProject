import cv2
import numpy as np
from ais_bench.infer.interface import InferSession  # 昇腾推理接口

# 1. 初始化NPU模型
model = InferSession(device_id=0, model_path="./runs/train/train6/weights/best.om")


# 2. 图像预处理（适配YOLOv11输入）
def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # 添加批次维度+归一化
    return img


# 3. NPU推理
def infer_npu(image):
    blob = preprocess(image)
    outputs = model.infer(blob, mode="static")  # static模式优化延迟[1](@ref)
    return outputs


# 4. 后处理（解析YOLOv11输出）
def postprocess(outputs, conf_thres=0.5, iou_thres=0.45):
    # YOLOv11输出格式: [1,116,8400] (检测头) + [1,32,160,160] (分割头)[11](@ref)
    detections = outputs[0][0].transpose(1, 0)  # 转置为[8400,116]
    masks = outputs[1][0]  # [32,160,160]

    # 过滤低置信度框
    keep = detections[:, 4] > conf_thres
    detections = detections[keep]

    # NMS处理（使用昇腾加速算子）
    from aclop import nms  # 昇腾专用NMS算子[1](@ref)
    boxes = detections[:, :4]
    scores = detections[:, 4]
    keep = nms(boxes, scores, iou_threshold=iou_thres)
    return detections[keep], masks