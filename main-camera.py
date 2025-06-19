import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time

# ====================== 配置参数 ======================
MODEL_PATH = "./runs/train/train6/weights/best.om"
VIDEO_PATH = "./datasets/test/30386095338-1-192.mp4"
CLASS_NAMES = ["buoy"]  # 类别名称
CONF_THRESH = 0.3  # 置信度阈值
NMS_THRESH = 0.35  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸
SHOW_WINDOW = True  # 是否显示实时检测窗口
USE_CAMERA = True  # 使用USB摄像头还是视频文件
CAMERA_INDEX = 0  # USB摄像头设备索引
RESOLUTION = (1280, 720)  # 摄像头分辨率设置
DEBUG_MODE = False  # 调试模式

# 全局配置
global_aspect_ratio = True  # 宽高比处理开关


def preprocess(frame):
    """改进的图像预处理（支持宽高比保持）"""
    # 记录原始尺寸
    orig_h, orig_w = frame.shape[:2]

    if global_aspect_ratio:
        # 计算缩放比例并添加灰边（修复比例计算）
        scale = min(INPUT_SIZE[0] / orig_w, INPUT_SIZE[1] / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # 创建填充图像
        img = np.full((INPUT_SIZE[1], INPUT_SIZE[0], 3), 114, dtype=np.uint8)
        y_offset = (INPUT_SIZE[1] - new_h) // 2
        x_offset = (INPUT_SIZE[0] - new_w) // 2
        img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # 保存转换参数（增加缩放因子）
        meta = {
            'orig_shape': (orig_h, orig_w),
            'resized_shape': (new_h, new_w),
            'offset': (x_offset, y_offset),
            'scale': scale,
            'pad_w': INPUT_SIZE[0] - new_w,
            'pad_h': INPUT_SIZE[1] - new_h
        }
    else:
        # 直接缩放
        img = cv2.resize(frame, INPUT_SIZE)
        meta = {'orig_shape': (orig_h, orig_w)}

    # 转换颜色空间和调整尺寸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    return img, meta


def postprocess(outputs, meta):
    """改进的后处理（精确坐标还原）"""
    # 获取模型输出[1,84,8400] -> [8400,84]
    predictions = np.squeeze(outputs[0])
    if predictions.ndim == 2:
        predictions = predictions.transpose((1, 0))
    else:
        predictions = predictions.reshape(84, -1).transpose((1, 0))

    # 分离边界框和类别分数
    boxes = predictions[:, :4]  # [cx, cy, w, h]
    scores = predictions[:, 4:4 + len(CLASS_NAMES)]  # 类别分数

    # 转换边界框格式 (cx,cy,w,h) -> (x1,y1,x2,y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes = np.column_stack((x1, y1, x2, y2))

    # 获取原始参数
    orig_h, orig_w = meta['orig_shape']

    # 坐标还原（考虑灰边偏移）
    if global_aspect_ratio and 'offset' in meta:
        x_offset, y_offset = meta['offset']
        scale = meta['scale']

        # 去除填充偏移并缩放回原始尺寸
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_offset) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_offset) / scale
    else:
        # 直接缩放
        boxes[:, [0, 2]] *= orig_w / INPUT_SIZE[0]
        boxes[:, [1, 3]] *= orig_h / INPUT_SIZE[1]

    # 边界检查（防止坐标越界）
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)  # y3

    # 应用NMS（使用改进的索引处理）
    detections = []
    if boxes.shape[0] > 0:
        # 获取最大类别分数和类别ID
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # 应用OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            max_scores.tolist(),
            CONF_THRESH,
            NMS_THRESH
        )

        if indices is not None:
            indices = indices.flatten()
            for idx in indices:
                confidence = max_scores[idx]
                if confidence > CONF_THRESH:
                    class_id = class_ids[idx]
                    detections.append({
                        "class": CLASS_NAMES[class_id],
                        "confidence": float(confidence),
                        "box": [int(x) for x in boxes[idx]]
                    })
    return detections


def main():
    global global_aspect_ratio

    # 初始化模型
    session = InferSession(device_id=0, model_path=MODEL_PATH)

    # 打开视频源
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"使用USB摄像头 (设备索引: {CAMERA_INDEX})，分辨率: {RESOLUTION[0]}x{RESOLUTION[1]}")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"处理视频文件: {VIDEO_PATH}")

    if not cap.isOpened():
        print(f"错误: 无法打开视频源")
        return

    frame_count = 0
    start_time = time.time()
    fps_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if USE_CAMERA:
                print("摄像头读取错误，尝试重新获取帧...")
                time.sleep(0.1)
                continue
            else:
                break

        # 摄像头图像水平翻转
        if USE_CAMERA:
            frame = cv2.flip(frame, 1)

        # 记录帧开始时间
        frame_start = time.time()
        frame_count += 1

        # 预处理
        blob, meta = preprocess(frame)

        # 推理
        outputs = session.infer([blob])

        # 后处理
        detections = postprocess(outputs, meta)

        # 计算帧处理时间
        frame_time = time.time() - frame_start
        fps_history.append(1.0 / (frame_time + 1e-6))
        avg_fps = np.mean(fps_history[-30:])  # 最近30帧的平均FPS

        # 打印时间信息
        print(f"帧 {frame_count}: {len(detections)}个目标 | FPS: {avg_fps:.1f}")

        # 在图像上绘制结果
        display_frame = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']

            # 确保坐标有效
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(display_frame.shape[1] - 1, x2), min(
                display_frame.shape[0] - 1, y2)

            # 绘制边界框（增加颜色对比度）
            color = (0, 255, 0)  # 绿色
            thickness = 2
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # 绘制文本背景（提高可读性）
            label = f"{det['class']}: {det['confidence']:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness=1)[0]
            cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 5),
                          (x1 + text_size[0], y1), color, -1)

            # 绘制文本
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 调试模式下标记角点
            if DEBUG_MODE:
                cv2.circle(display_frame, (x1, y1), 5, (0, 0, 255), -1)
                cv2.circle(display_frame, (x2, y2), 5, (255, 0, 0), -1)

        # 显示性能信息
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mode_text = f"Mode: {'Camera' if USE_CAMERA else 'Video'}"
        cv2.putText(display_frame, mode_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        aspect_text = f"Aspect: {'On' if global_aspect_ratio else 'Off'}"
        cv2.putText(display_frame, aspect_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # 显示窗口
        if SHOW_WINDOW:
            cv2.imshow('Detection', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                global_aspect_ratio = not global_aspect_ratio
                print(f"切换宽高比模式: {'保持宽高比' if global_aspect_ratio else '直接缩放'}")

    # 性能统计
    total_duration = time.time() - start_time
    avg_fps = frame_count / total_duration

    print("\n" + "=" * 50)
    print(f"总帧数: {frame_count} | 总耗时: {total_duration:.2f}s | 平均FPS: {avg_fps:.2f}")
    print("=" * 50)

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()