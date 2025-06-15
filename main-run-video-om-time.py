import cv2
import numpy as np
import time
import os
from datetime import datetime
from tqdm import tqdm
from ais_bench.infer.interface import InferSession


class NPUInferencer:
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = InferSession(device_id=0, model_path=model_path)
        self.conf_threshold = conf_threshold
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_shape = self.model.get_outputs()[0].shape
        self.total_inference_time = 0
        self.frame_count = 0

        # 创建视频保存目录
        self.video_dir = "./mp4"
        os.makedirs(self.video_dir, exist_ok=True)

        # 视频分段参数
        self.segment_interval = 20  # 每20秒保存一个视频片段
        self.segment_start_time = 0  # 视频时间而非实时时间
        self.segment_count = 0
        self.out_video = None
        self.current_video_path = ""
        self.fps = 0  # 视频帧率
        self.frame_index = 0  # 当前帧在视频中的位置

        # 类别映射表（根据实际模型修改）
        self.class_names = {
            0: "buoy"
            # 添加更多类别...
        }

        # 检测结果记录
        self.detection_log = []

    def preprocess(self, frame):
        start_time = time.perf_counter()
        img = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)[np.newaxis]
        img = img.astype("float32") / 255.0
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"预处理耗时: {elapsed:.2f}ms")
        return img

    def postprocess(self, outputs, frame):
        start_time = time.perf_counter()
        h, w = frame.shape[:2]

        # 存储检测到的目标
        detected_objects = []

        # 确定输出格式
        if outputs[0].shape == (1, 8400, 84):
            detections = outputs[0][0]
        else:
            detections = outputs[0][0].transpose(1, 0)

        if detections.size == 0:
            return frame, detected_objects

        for row in detections:
            if len(row) < 6:
                continue
            try:
                x1, y1, x2, y2, conf, cls_id = row[:6]
                if conf < self.conf_threshold:
                    continue

                # 转换为图像坐标
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

                # 获取类别名称
                class_id = int(cls_id)
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                # 计算中心点坐标
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 存储检测信息
                obj_info = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "frame_index": self.frame_index,
                    "timestamp": self.frame_index / self.fps
                }
                detected_objects.append(obj_info)

                # 绘制边界框和标签
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # 绘制中心点
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except (ValueError, IndexError) as e:
                print(f"处理检测结果时出错: {e}")
                continue

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"后处理耗时: {elapsed:.2f}ms")
        return frame, detected_objects

    def log_detection(self, detected_objects):
        """记录并打印检测到的目标信息"""
        if not detected_objects:
            return

        # 按类别分组统计
        class_counts = {}
        for obj in detected_objects:
            class_name = obj["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # 记录检测日志
            self.detection_log.append(obj)

        # 打印检测摘要
        timestamp = datetime.now().strftime("%H:%M:%S")
        time_in_video = self.frame_index / self.fps
        print(f"\n🔔 检测到目标 (视频时间: {time_in_video:.1f}s, 系统时间: {timestamp})")

        # 打印每个类别的统计信息
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count}个")

        # 打印置信度最高的目标
        highest_conf_obj = max(detected_objects, key=lambda x: x["confidence"])
        print(f"  最高置信度目标: {highest_conf_obj['class_name']} "
              f"{highest_conf_obj['confidence']:.2%} "
              f"位置: ({highest_conf_obj['center'][0]}, {highest_conf_obj['center'][1]})")

    def init_video_writer(self, width, height, fps):
        """初始化视频写入器"""
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_path = os.path.join(
            self.video_dir,
            f"segment_{self.segment_count}_{timestamp}.mp4"
        )

        # 设置视频编码器（优先使用H.264编码）
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264编码
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4编码
            cv2.VideoWriter_fourcc(*'X264'),  # 开源H.264实现
            cv2.VideoWriter_fourcc(*'XVID')  # 兼容性编码器
        ]

        # 尝试不同编码器
        for fourcc in fourcc_options:
            self.out_video = cv2.VideoWriter(
                self.current_video_path,
                fourcc,
                fps,
                (width, height)
            )
            if self.out_video.isOpened():
                print(f"✅ 成功使用编码器初始化视频写入器: {self.current_video_path}")
                break
            else:
                print(f"⚠️ 编码器初始化失败，尝试下一个选项")

        # 如果所有编码器都失败，使用默认选项
        if not self.out_video.isOpened():
            self.current_video_path = os.path.join(
                self.video_dir,
                f"segment_{self.segment_count}_{timestamp}.avi"
            )
            self.out_video = cv2.VideoWriter(
                self.current_video_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (width, height)
            )
            print(f"⚠️ 使用兼容性编码器: {self.current_video_path}")

        # 重置分段计时
        self.segment_start_time = self.frame_index / self.fps
        self.segment_count += 1

    def predict(self, source, show=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")

        # 获取视频参数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频总帧数: {total_frames}, FPS: {self.fps:.1f}, 分辨率: {width}x{height}")

        # 初始化第一个视频写入器
        self.init_video_writer(width, height, self.fps)

        # 性能监控数据结构
        inference_times = []
        self.frame_index = 0

        for _ in tqdm(range(total_frames), desc="NPU推理进度"):
            ret, frame = cap.read()
            if not ret:
                break

            # 更新帧索引
            self.frame_index += 1

            # 完整处理流程计时
            frame_start = time.perf_counter()

            # 预处理
            blob = self.preprocess(frame)

            # NPU推理核心计时
            infer_start = time.perf_counter()
            outputs = self.model.infer(blob, mode="static")
            infer_elapsed = (time.perf_counter() - infer_start) * 1000

            # 记录并打印推理耗时
            inference_times.append(infer_elapsed)
            self.total_inference_time += infer_elapsed
            self.frame_count += 1
            print(f"NPU推理耗时: {infer_elapsed:.2f}ms")

            # 后处理（目标检测和标记）
            result_frame, detected_objects = self.postprocess(outputs, frame)

            # 记录并打印检测结果
            self.log_detection(detected_objects)

            # 帧处理总耗时
            frame_elapsed = (time.perf_counter() - frame_start) * 1000
            print(f"单帧总耗时: {frame_elapsed:.2f}ms")

            # 计算当前视频时间（秒）
            current_video_time = self.frame_index / self.fps

            # 检查是否需要创建新的视频片段
            if current_video_time - self.segment_start_time >= self.segment_interval:
                # 关闭当前视频文件
                if self.out_video is not None:
                    self.out_video.release()
                    print(f"🟢 视频片段已保存: {self.current_video_path}")

                # 初始化新的视频写入器
                self.init_video_writer(width, height, self.fps)

            # 将标记后的帧写入当前视频文件
            if self.out_video is not None and self.out_video.isOpened():
                self.out_video.write(result_frame)

            if show:
                # 在画面上显示片段信息
                segment_info = f"Segment: {self.segment_count} | Time: {current_video_time:.1f}s"
                cv2.putText(result_frame, segment_info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示检测计数
                if detected_objects:
                    detection_count = len(detected_objects)
                    detection_info = f"Detections: {detection_count}"
                    cv2.putText(result_frame, detection_info, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('NPU加速-YOLO', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 释放资源
        cap.release()
        if self.out_video is not None and self.out_video.isOpened():
            self.out_video.release()
            print(f"🟢 最后一个视频片段已保存: {self.current_video_path}")
        cv2.destroyAllWindows()

        # 性能总结报告
        if inference_times:
            avg_time = np.mean(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            print("\n" + "=" * 50)
            print(f"性能分析报告:")
            print(f"- 总处理帧数: {self.frame_count}")
            print(f"- 平均推理耗时: {avg_time:.2f}ms")
            print(f"- 最快推理耗时: {min_time:.2f}ms")
            print(f"- 最慢推理耗时: {max_time:.2f}ms")
            print(f"- 预估FPS: {1000 / avg_time:.1f} (仅推理)")
            print(f"- 保存视频片段数: {self.segment_count}")
            print(f"- 视频保存位置: {os.path.abspath(self.video_dir)}")

            # 检测结果汇总
            if self.detection_log:
                print("\n检测结果汇总:")
                class_counts = {}
                for obj in self.detection_log:
                    class_name = obj["class_name"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                for class_name, count in class_counts.items():
                    print(f"  - {class_name}: {count}次")

                # 打印首次和最后一次检测时间
                first_detection = min(self.detection_log, key=lambda x: x["timestamp"])
                last_detection = max(self.detection_log, key=lambda x: x["timestamp"])
                print(f"\n首次检测: {first_detection['class_name']} "
                      f"在 {first_detection['timestamp']:.1f}s")
                print(f"最后一次检测: {last_detection['class_name']} "
                      f"在 {last_detection['timestamp']:.1f}s")

            print("=" * 50)


if __name__ == '__main__':
    # 定义类别映射（根据实际模型修改）
    class_names = {
        0: "buoy"
    }

    inferencer = NPUInferencer(
        model_path="./runs/train/train6/weights/best.om",
        conf_threshold=0.3,
    )

    # 设置自定义类别映射
    inferencer.class_names = class_names

    inferencer.predict(
        source=r'./datasets/test/30386095338-1-192.mp4',
        show=False
    )
