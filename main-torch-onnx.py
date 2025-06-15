import torch
import onnx

if __name__ == '__main__':

    # 加载 PyTorch 模型（以 YOLO 为例）
    model = torch.load("runs/train/train6/weights/best.pt")["model"]  # YOLO 模型需提取 'model' 字段[1](@ref)
    model.eval()  # 切换推理模式
    model.float()  # FP32 精度（兼容 NPU）
    model.fuse()   # 融合 Conv+BN 层加速[1](@ref)

    # 设置输入参数（固定尺寸适配 NPU）
    input_shape = (1, 3, 640, 640)  # [batch, channel, height, width]
    dummy_input = torch.randn(input_shape)

    # 导出 ONNX（关键参数优化）
    torch.onnx.export(
        model,
        dummy_input,
        "best.onnx",  # 输出文件名
        export_params=True,       # 包含权重
        opset_version=13,        # 推荐 >=11（支持 NPU 算子）[3,7](@ref)
        do_constant_folding=False,  # 常量折叠优化（减少计算量）[6,7](@ref)
        input_names=["images"],    # 输入节点名（与 OM 匹配）
        output_names=["output"],   # 输出节点名（避免默认名冲突）
        dynamic_axes=None,        # **固定维度**（NPU 需静态 shape）[5](@ref)
    )
    print("✅ ONNX 导出完成")

    # 验证 ONNX 模型
    onnx_model = onnx.load("best.onnx")
    onnx.checker.check_model(onnx_model)  # 检查结构完整性[8](@ref)
