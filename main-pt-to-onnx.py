from ultralytics import YOLO



if __name__ == '__main__':
    # 加载预训练模型（替换为你的路径）
    model = YOLO("runs/train/train6/weights/best.pt")

    # 导出ONNX格式[1,2,3,6,9](@ref)
    model.export(
        format="onnx",  # 输出格式
        imgsz=640,  # 输入尺寸（与训练一致）
        opset=13,  # ONNX算子集版本（推荐12+）
        simplify=True,  # 启用模型简化（移除冗余节点）
        dynamic=True,  # 支持动态输入尺寸（如batch/分辨率）
        batch=1  # 指定默认batch_size
    )