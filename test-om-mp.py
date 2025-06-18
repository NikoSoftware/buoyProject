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
    
    # 提取有效检测结果
    detections = []
    if indices is not None:
        for i in indices:
            # 修复索引访问问题
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
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {VIDEO_PATH}")
        return
    
    frame_count = 0
    total_preprocess = 0.0
    total_inference = 0.0
    total_postprocess = 0.0
    total_frame = 0.0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # ====================== 帧开始计时 ======================
        frame_start = time.time()
        frame_count += 1
        
        # 1. 预处理计时
        preprocess_start = time.time()
        orig_h, orig_w = frame.shape[:2]
        blob = preprocess(frame)
        preprocess_time = time.time() - preprocess_start
        
        # 2. 推理计时
        inference_start = time.time()
        outputs = session.infer([blob])
        inference_time = time.time() - inference_start
        
        # 3. 后处理计时
        postprocess_start = time.time()
        detections = postprocess(outputs, (orig_h, orig_w))
        postprocess_time = time.time() - postprocess_start
        
        # 4. 帧总处理时间
        frame_time = time.time() - frame_start
        
        # 累计时间统计
        total_preprocess += preprocess_time
        total_inference += inference_time
        total_postprocess += postprocess_time
        total_frame += frame_time
        
        # ====================== 打印时间信息 ======================
        print(f"\n帧 {frame_count} 时间统计:")
        print(f"预处理: {preprocess_time*1000:.2f}ms | "
              f"推理: {inference_time*1000:.2f}ms | "
              f"后处理: {postprocess_time*1000:.2f}ms | "
              f"总帧耗时: {frame_time*1000:.2f}ms")
        
        # 打印检测结果
        print(f"检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            print(f"  目标 {i+1}: {det['class']} | "
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
    
    # ====================== 最终性能统计 ======================
    end_time = time.time()
    total_duration = end_time - start_time
    fps = frame_count / total_duration
    
    print("\n" + "="*50)
    print("最终性能统计:")
    print(f"总帧数: {frame_count} | 总耗时: {total_duration:.2f}s | 平均FPS: {fps:.2f}")
    print(f"平均预处理时间: {total_preprocess/frame_count*1000:.2f}ms/帧")
    print(f"平均推理时间: {total_inference/frame_count*1000:.2f}ms/帧")
    print(f"平均后处理时间: {total_postprocess/frame_count*1000:.2f}ms/帧")
    print(f"平均帧处理时间: {total_frame/frame_count*1000:.2f}ms/帧")
    print("="*50)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
