from ultralytics import YOLO

def export_yolov11_onnx():
    """ 使用官方API导出ONNX（兼容YOLOv11n）"""
    # 1. 加载训练好的模型
    model = YOLO("runs/train/train6/weights/best.pt")
    
    # 2. 原生导出ONNX（关键参数优化）
    model.export(
        format="onnx",
        imgsz=640,              # 固定输入尺寸（NPU必需）
        opset=13,                # 算子集版本≥12
        simplify=True,           # 启用模型简化（移除冗余算子）
        dynamic=False,           # 静态输入（NPU部署要求）
        batch=1,                 # 批处理大小
        device="cpu",            # 导出设备
        nms=False                # 不内置NMS（需自行实现后处理）
    )
    print("✅ ONNX导出成功！输出文件：best.onnx")

if __name__ == '__main__':
    export_yolov11_onnx()
