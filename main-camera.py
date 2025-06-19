import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
import time

# ====================== 配置参数 ======================
MODEL_PATH = "./runs/train/train6/weights/best.om"
VIDEO_PATH = "./datasets/test/30386095338-1-192.mp4"
CLASS_NAMES = ["buoy"]  # 根据你的buoy.yaml修改类别名称
CONF_THRESH = 0.3  # 置信度阈值
NMS_THRESH = 0.35  # NMS阈值
INPUT_SIZE = (640, 640)  # 模型输入尺寸
SHOW_WINDOW = True  # 控制是否显示实时检测窗口
USE_CAMERA = True  # 设置为True使用USB摄像头，False使用视频文件
CAMERA_INDEX = 0  # USB摄像头设备索引，默认为0
RESOLUTION = (1280, 720)  # 摄像头分辨率设置
ASPECT_RATIO = True  # 保持宽高比进行缩放，解决位置漂移问题
DEBUG_MODE = False  # 调试模式，显示坐标转换信息

# 声明为全局变量（修复SyntaxError的关键）
global_aspect_ratio = ASPECT_RATIO


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

        # 保存转换参数
        meta = {
            'orig_shape': (orig_h, orig_w),
            'resized_shape': (new_h, new_w),
            'offset': (x_offset, y_offset),
            'scale': scale
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

        # 应用OpenCV NMS - 修复索引处理问题
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            max_scores.tolist(),
            CONF_THRESH,
            NMS_THRESH
        )

        # 处理不同OpenCV版本返回的索引格式问题
        if indices is not None:
            # OpenCV 4.x返回元组，OpenCV 3.x返回numpy数组
            if isinstance(indices, tuple) or isinstance(indices, list):
                indices = np.array(indices, dtype=np.int32).flatten()
            else:
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
    total_preprocess = 0.0
    total_inference = 0.0
    total_postprocess = 0.0
    total_frame = 0.0
    start_time = time.time()

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

        # ====================== 帧开始计时 ======================
        frame_start = time.time()
        frame_count += 1

        # 1. 预处理计时（现在返回预处理图像和元数据）
        preprocess_start = time.time()
        blob, meta = preprocess(frame)
        preprocess_time = time.time() - preprocess_start

        # 2. 推理计时
        inference_start = time.time()
        outputs = session.infer([blob])
        inference_time = time.time() - inference_start

        # 3. 后处理计时（传入meta数据）
        postprocess_start = time.time()
        detections = postprocess(outputs, meta)
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
        print(f"预处理: {preprocess_time * 1000:.2f}ms | "
              f"推理: {inference_time * 1000:.2f}ms | "
              f"后处理: {postprocess_time * 1000:.2f}ms | "
              f"总帧耗时: {frame_time * 1000:.2f}ms")

        # 打印检测结果
        print(f"检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            print(f"  目标 {i + 1}: {det['class']} | "
                  f"置信度: {det['confidence']:.4f} | "
                  f"位置: [{det['box'][0]}, {det['box'][1]}, {det['box'][2]}, {det['box'][3]}]")

            # 在图像上绘制结果
            x1, y1, x2, y2 = det['box']
            # 确保坐标有效
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

            # 绘制边界框（增加颜色对比度）
            color = (0, 255, 0)  # 绿色
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 绘制文本背景（提高可读性）
            label = f"{det['class']}: {det['confidence']:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness=1)

            # 智能调整文本位置（避免超出图像边界）
            text_y = y1 - 5
            if text_y < label_height + 5:  # 如果上方空间不足
                text_y = y2 + label_height + 5  # 显示在框下方

            # 绘制文本背景矩形
            cv2.rectangle(frame,
                          (x1, text_y - label_height - 5),
                          (x1 + label_width, text_y + 5),
                          color, -1)

            # 绘制文本
            cv2.putText(frame, label,
                        (x1, text_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 调试模式下标记角点
            if DEBUG_MODE:
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)  # 左上角红色
                cv2.circle(frame, (x2, y2), 5, (255, 0, 0), -1)  # 右下角蓝色
                print(f"原始尺寸: {meta['orig_shape']}, 转换后坐标: {det['box']}")

        # 显示帧率信息
        fps_text = f"FPS: {1 / (frame_time + 1e-6):.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示当前模式
        mode_text = f"Mode: {'Camera' if USE_CAMERA else 'Video'}"
        cv2.putText(frame, mode_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        aspect_text = f"Aspect: {'On' if global_aspect_ratio else 'Off'}"
        cv2.putText(frame, aspect_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # ====================== 显示控制开关 ======================
        if SHOW_WINDOW:
            # 显示实时结果
            window_title = 'USB Camera Detection' if USE_CAMERA else 'Video Detection'
            cv2.imshow(window_title, frame)

            # 检查退出按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停/继续
                while True:
                    key = cv2.waitKey(1)
                    if key == ord(' '):
                        break
                    elif key == ord('q'):
                        cv2.destroyAllWindows()
                        cap.release()
                        return
            elif key == ord('a'):  # 切换宽高比模式
                global_aspect_ratio = not global_aspect_ratio
                print(f"切换宽高比模式: {'保持宽高比' if global_aspect_ratio else '直接缩放'}")
        else:
            # 无头模式下，仅延时1ms保持处理节奏
            time.sleep(0.001)

    # ====================== 最终性能统计 ======================
    end_time = time.time()
    total_duration = end_time - start_time
    fps = frame_count / total_duration

    print("\n" + "=" * 50)
    print("最终性能统计:")
    print(f"总帧数: {frame_count} | 总耗时: {total_duration:.2f}s | 平均FPS: {fps:.2f}")
    print(f"平均预处理时间: {total_preprocess / frame_count * 1000:.2f}ms/帧")
    print(f"平均推理时间: {total_inference / frame_count * 1000:.2f}ms/帧")
    print(f"平均后处理时间: {total_postprocess / frame_count * 1000:.2f}ms/帧")
    print(f"平均帧处理时间: {total_frame / frame_count * 1000:.2f}ms/帧")
    print("=" * 50)

    # 清理资源
    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()