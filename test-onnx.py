
from ultralytics import YOLO

# 使用示例
if __name__ == '__main__':
    model = YOLO('./runs/train/train6/weights/best.onnx')
    results = model.predict('./datasets/test/30386095338-1-192.mp4', save=True)
    print("官方验证结果:", results[0].boxes)
