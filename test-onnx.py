import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime

class ONNXInferencer:
    def __init__(self, model_path, conf_threshold=0.3):
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1,3,H,W]
        self.conf_threshold = conf_threshold
        self.class_names = {0: "buoy"}  # 根据实际模型修改
        
        print(f"ONNX模型加载成功 | 输入尺寸: {self.input_shape[3]}x{self.input_shape[2]}")

    def preprocess(self, frame):
        """调整尺寸+归一化+转NCHW格式"""
        img = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return img

    def postprocess(self, outputs, frame):
        """解析YOLOv8输出格式 [1,84,8400]"""
        detections = outputs[0].transpose(0, 2, 1)  # 转置为[1,8400,84]
        return self._parse_detections(detections[0], frame)

    def _parse_detections(self, detections, frame):
        """提取边界框和类别信息"""
        height, width = frame.shape[:2]
        detected_objects = []
        
        for detection in detections:
            scores = detection[4:]  # 各类别置信度
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence < self.conf_threshold:
                continue
                
            # 解析边界框 (cx, cy, w, h) → (x1, y1, x2, y2)
            cx, cy, w, h = detection[:4] * [width, height, width, height]
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            
            # 存储结果
            obj_info = {
                "class_id": class_id,
                "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "center": (int(cx), int(cy))
            }
            detected_objects.append(obj_info)
            
            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj_info['class_name']} {confidence:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, detected_objects

    def predict(self, source, show=True):
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            # 预处理 → 推理 → 后处理
            blob = self.preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: blob})
            result_frame, detections = self.postprocess(outputs, frame.copy())
            
            if show:
                cv2.imshow('ONNX Inference', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        cv2.destroyAllWindows()

# 使用示例
if __name__ == '__main__':
    inferencer = ONNXInferencer(
        model_path="./runs/train/train6/weights/best.onnx",
        conf_threshold=0.3
    )
    inferencer.predict(source='./datasets/test/30386095338-1-192.mp4', show=False)
