from ultralytics import YOLO
import cv2
import time
from datetime import datetime

def run():
    # 源文件设置
    source = r'./datasets/test/30386095338-1-192.mp4'  # 视频文件
    # source = r'./datasets/test/20250611213828.png'    # 图片文件
    
    # 加载模型
    model = YOLO("./runs/train/train6/weights/best.pt")
    
    # 执行预测（使用stream=True处理视频流）
    results = model.predict(
        source=source,
        conf=0.3,
        stream=True,  # 流式处理视频
        show=True,     # 显示检测结果
        verbose=False  # 关闭默认输出
    )
    
    # 处理每一帧的检测结果
    for frame_idx, result in enumerate(results):
        # 获取检测数据
        boxes = result.boxes
        detection_data = []
        
        # 提取检测信息
        for box in boxes:
            # 边界框坐标 (xyxy格式)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 置信度和类别
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            
            # 中心点坐标
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 存储检测数据
            obj_info = {
                "frame": frame_idx,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "timestamp": time.time()  # 当前系统时间戳
            }
            detection_data.append(obj_info)
        
        # 打印检测数据
        print_detections(detection_data, frame_idx)

def print_detections(detections, frame_idx):
    """格式化打印检测结果"""
    if not detections:
        print(f"帧 {frame_idx}: 未检测到目标")
        return
    
    # 按类别分组
    class_summary = {}
    for det in detections:
        class_name = det['class_name']
        class_summary[class_name] = class_summary.get(class_name, 0) + 1
    
    # 打印摘要
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n帧 {frame_idx} | 系统时间: {timestamp}")
    print("=" * 50)
    print(f"检测到 {len(detections)} 个目标:")
    
    # 打印每个目标的详细信息
    for i, det in enumerate(detections):
        print(f"{i+1}. {det['class_name']} (ID:{det['class_id']})")
        print(f"   置信度: {det['confidence']:.2%}")
        print(f"   边界框: ({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}) → ({det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})")
        print(f"   中心点: ({det['center'][0]}, {det['center'][1]})")
    
    # 打印类别统计
    print("\n类别统计:")
    for cls, count in class_summary.items():
        print(f"  - {cls}: {count}个")
    print("=" * 50)

if __name__ == '__main__':
    run()
