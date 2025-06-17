import cv2
import numpy as np
import onnxruntime as ort  # ONNX推理引擎


def run():
    # 模型和媒体路径配置
    #source = r'./datasets/test/30386095338-1-192.mp4'  # 视频文件
    source = r'./datasets/test/20250611213828.png'    # 图片文件
    onnx_model_path = "./runs/train/train6/weights/best.onnx"  # ONNX模型路径

    # 1. 加载ONNX模型[8,9](@ref)
    session = ort.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider']  # 使用CPU推理
    )

    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. 媒体处理（支持图片和视频）
    if source.endswith(('.png', '.jpg', '.jpeg')):
        # 图片处理
        frame = cv2.imread(source)
        if frame is None:
            print(f"错误：无法读取图片 {source}")
            return

        # 预处理并推理
        blob = preprocess(frame)
        outputs = session.run([output_name], {input_name: blob})[0]

        # 后处理并显示
        result_frame = postprocess(outputs, frame)
        cv2.imshow("YOLOv11 ONNX检测结果", result_frame)
        cv2.waitKey(0)

    else:
        # 视频处理
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"错误：无法打开视频 {source}")
            return

        # 实时处理每一帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理并推理
            blob = preprocess(frame)
            outputs = session.run([output_name], {input_name: blob})[0]

            # 后处理并显示
            result_frame = postprocess(outputs, frame)
            cv2.imshow("YOLOv11 ONNX实时检测", result_frame)

            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break

        cap.release()

    cv2.destroyAllWindows()


def preprocess(frame):
    """图像预处理函数[2,5](@ref)"""
    # 调整尺寸 (640x640)
    resized = cv2.resize(frame, (640, 640))

    # 通道转换 BGR→RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 归一化 (0-1范围)
    normalized = rgb.astype(np.float32) / 255.0

    # 维度转换 HWC→CHW 并添加批次维度
    chw = normalized.transpose(2, 0, 1)
    blob = np.expand_dims(chw, axis=0)

    return blob


def postprocess(outputs, orig_frame):
    """后处理函数[4,6](@ref)"""
    # YOLOv11输出格式: [1, 84, 8400]
    detections = outputs[0].transpose()  # 转置为[8400, 84]
    orig_h, orig_w = orig_frame.shape[:2]

    for det in detections:
        # 解析坐标 (cx, cy, w, h)
        cx, cy, w, h = det[:4]

        # 解析置信度
        scores = det[4:84]
        confidence = np.max(scores)

        # 置信度过滤
        if confidence < 0.1:  # 与原始conf参数一致
            continue

        # 获取类别ID
        class_id = np.argmax(scores)

        # 坐标转换 (cx,cy,w,h) → (x1,y1,x2,y2)并缩放到原始尺寸
        x1 = int((cx - w / 2) * orig_w)
        y1 = int((cy - h / 2) * orig_h)
        x2 = int((cx + w / 2) * orig_w)
        y2 = int((cy + h / 2) * orig_h)

        # 绘制检测框
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加标签和置信度
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(orig_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return orig_frame


if __name__ == '__main__':
    run()