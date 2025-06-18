import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time

# 配置参数
MODEL_PATH = "./runs/train/train6/weights/best.om"
VIDEO_PATH = "./datasets/test/30386095338-1-192.mp4"
CLASS_NAMES = ["buoy"]  # 根据你的buoy.yaml修改类别名称
CONF_THRESH = 0.3  # 置信度阈值
NMS_THRESH = 0.35  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸

def preprocess(frame):
    """图像预处理"""
    # 转换颜色空间和调整尺寸
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    
    # 归一化并转换格式
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # 添加batch维度
    return img

def postprocess(outputs, orig_shape):
    """后处理：解析检测结果"""
    # YOLOv11输出格式: [1, 84, 8400]
    predictions = np.squeeze(outputs[0])  # 移除batch维度
    predictions = predictions.transpose((1, 0))  # 转置为[8400, 84]
    
    # 获取边界框和类别分数
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    
    # 转换边界框格式 (cx,cy,w,h) -> (x1,y1,x2,y2)
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]       # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]       # y2
    
    # 缩放回原始图像尺寸
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] *= w / INPUT_SIZE[0]
    boxes[:, [1, 3]] *= h / INPUT_SIZE[1]
    
    # 应用NMS - 修复索引访问问题
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), 
        np.max(scores, axis=1).tolist(),
        CONF_THRESH, 
        NMS_THRESH
    )
    
    # 提取有效检测结果 - 修复部分
    detections = []
    for i in indices:
        # 修复索引访问问题：处理numpy.int64类型
        idx = int(i) if hasattr(i, '__int__') else (i[0] if isinstance(i, (list, tuple, np.ndarray)) else i)
        
        class_id = np.argmax(scores[idx])
        confidence = scores[idx][class_id]
        if confidence > CONF_THRESH:
            detections.append({
                "class": CLASS_NAMES[class_id],
                "confidence": float(confidence),
                "box": [int(x) for x in boxes[idx]]
            })
    return detections

def main():
    # 初始化模型
    session = InferSession(device_id=0, model_path=MODEL_PATH)
    
    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 预处理
        orig_h, orig_w = frame.shape[:2]
        blob = preprocess(frame)
        
        # 推理
        outputs = session.infer([blob])
        
        # 后处理
        detections = postprocess(outputs, (orig_h, orig_w))
        
        # 打印检测信息
        frame_count += 1
        print(f"\n帧 {frame_count} 检测结果:")
        for i, det in enumerate(detections):
            print(f"目标 {i+1}: {det['class']} | "
                  f"置信度: {det['confidence']:.4f} | "
                  f"位置: [{det['box'][0]}, {det['box'][1]}, {det['box'][2]}, {det['box'][3]}]")
            
            # 在图像上绘制结果
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 
                        f"{det['class']}: {det['confidence']:.2f}", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
        
        # 显示实时结果
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 性能统计
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"\n统计信息: 总帧数={frame_count} | 平均FPS={fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





