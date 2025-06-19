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
ASPECT_RATIO = True  # 保持宽高比进行缩放，解决位置漂移问题[1,2](@ref)
DEBUG_MODE = False  # 调试模式，显示坐标转换信息


def preprocess(frame):
    """改进的图像预处理（支持宽高比保持）"""
    # 记录原始尺寸
    orig_h, orig_w = frame.shape[:2]

    if ASPECT_RATIO:
        # 计算缩放比例并添加灰边
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
        # 直接缩放（原始逻辑）
        img = cv2.resize(frame, INPUT_SIZE)
        meta = {'orig_shape': (orig_h, orig_w)}

    # 转换颜色空间和调整尺寸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    return img, meta


def postprocess(outputs, meta):
    """改进的后处理（支持宽高比还原）"""
    # YOLOv11输出格式: [1, 84, 8400]
    predictions = np.squeeze(outputs[0])  # 移除batch维度
    predictions = predictions.transpose((1, 0))  # 转置为[8400, 84]

    # 获取边界框和类别分数
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]

    # 转换边界框格式 (cx,cy,w,h) -> (x1,y1,x2,y2)
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

    # 获取原始参数
    orig_h, orig_w = meta['orig_shape']

    if ASPECT_RATIO and 'offset' in meta:
        # 还原带灰边的坐标转换
        x_offset, y_offset = meta['offset']
        scale = meta['scale']

        # 去除填充偏移并缩放回原始尺寸
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_offset) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_offset) / scale
    else:
        # 直接缩放（原始逻辑）
        boxes[:, [0, 2]] *= orig_w / INPUT_SIZE[0]
        boxes[:, [1, 3]] *= orig_h / INPUT_SIZE[1]

    # 边界检查（防止坐标越界）[3](@ref)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)  # y3

    # 应用NMS
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

    # 打开视频源（USB摄像头或视频文件）
    if USE_CAMERA:
        # 打开USB摄像头
        cap = cv2.VideoCapture(CAMERA_INDEX)

        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

        # 设置帧率（可选）
        cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"正在使用USB摄像头 (设备索引: {CAMERA_INDEX})，分辨率: {RESOLUTION[0]}x{RESOLUTION[1]}")
    else:
        # 打开视频文件
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"正在处理视频文件: {VIDEO_PATH}")

    # 检查视频源是否成功打开
    if not cap.isOpened():
        if USE_CAMERA:
            print(f"错误: 无法打开USB摄像头 (设备索引: {CAMERA_INDEX})")
            print("可能原因: 1) 设备索引错误 2) 摄像头被其他程序占用 3) 驱动程序问题")
        else:
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
            if USE_CAMERA:
                # 对于摄像头，错误通常是临时性的，可以继续尝试
                print("摄像头读取错误，尝试重新获取帧...")
                time.sleep(0.1)
                continue
            else:
                # 对于视频文件，读取结束则退出
                break

        # 摄像头图像水平翻转（解决镜像问题）
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

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 智能调整文本位置（避免超出图像边界）[4](@ref)
            text_y = y1 - 10
            if text_y < 10:  # 如果上方空间不足
                text_y = y2 + 20  # 显示在框下方

            # 绘制类别和置信度
            cv2.putText(frame,
                        f"{det['class']}: {det['confidence']:.2f}",
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # 调试模式下显示坐标转换信息
            if DEBUG_MODE:
                print(f"原始尺寸: {meta['orig_shape']}, 转换后坐标: {det['box']}")
                # 标记边界框左上角
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)

        # 显示帧率信息
        fps_text = f"FPS: {1 / (frame_time + 1e-6):.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示当前模式
        mode_text = f"Mode: {'Camera' if USE_CAMERA else 'Video'}, Aspect: {'On' if ASPECT_RATIO else 'Off'}"
        cv2.putText(frame, mode_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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
                global ASPECT_RATIO
                ASPECT_RATIO = not ASPECT_RATIO
                print(f"切换宽高比模式: {'保持宽高比' if ASPECT_RATIO else '直接缩放'}")
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