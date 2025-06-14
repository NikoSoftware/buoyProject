import cv2
import numpy as np
import time
import os
from datetime import datetime
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
        self.video_dir = ".video"
        os.makedirs(self.video_dir, exist_ok=True)

        # 类别映射表
        self.class_names = {
            0: "buoy",  # 假设红色小瓶子是类别0
            # 添加更多类别如: 1: "person", 2: "car"...
        }

        # 视频分段参数
        self.segment_interval = 20  # 每20秒保存一个视频片段
        self.segment_start_time = time.time()
        self.segment_count = 0
        self.out_video = None
        self.current_video_path = ""
        self.actual_fps = 30.0  # 默认帧率

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

        # 确定输出格式
        if outputs[0].shape == (1, 8400, 84):
            detections = outputs[0][0]
        else:
            detections = outputs[0][0].transpose(1, 0)

        if detections.size == 0:
            return frame

        detected_objects = []  # 存储检测到的物体信息
        object_count = 0  # 检测到的物体计数器

        for row in detections:
            if len(row) < 6:
                continue
            try:
                x1, y1, x2, y2, conf, cls_id = row[:6]
                if conf < self.conf_threshold:
                    continue

                # 转换为图像坐标
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2  # 质心坐标

                # 获取类别名称
                class_name = self.class_names.get(int(cls_id), f"class_{int(cls_id)}")

                # 打印检测信息
                print(f"[Frame {self.frame_count}] 检测到目标: "
                      f"{class_name} (置信度: {conf:.2%}), "
                      f"边界框: [{x1}, {y1}, {x2}, {y2}], "
                      f"质心位置: ({x_c}, {y_c})")

                # 存储检测信息
                detected_objects.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": (x_c, y_c)
                })
                object_count += 1

                # 绘制边界框和标签
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x_c, y_c), 5, (0, 0, 255), -1)  # 绘制质心
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except (ValueError, IndexError) as e:
                print(f"处理检测结果时出错: {e}")
                continue

        # 打印本帧检测统计
        if object_count > 0:
            print(f"=== 帧 {self.frame_count} 检测统计 ===")
            print(f"检测到目标总数: {object_count}")
            print(f"主要目标: {detected_objects[0]['class']} "
                  f"(置信度: {detected_objects[0]['confidence']:.2%})")
            print("=" * 40)

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"后处理耗时: {elapsed:.2f}ms")
        return frame

    def init_video_writer(self, width, height, fps):
        """初始化视频写入器"""
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_path = os.path.join(
            self.video_dir,
            f"segment_{self.segment_count}_{timestamp}.mp4"  # 修改为MP4格式[1,3](@ref)
        )

        # 使用兼容性更好的H.264编码器[5](@ref)
        # 尝试多种编码器确保兼容性
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # 首选H.264编码[5](@ref)
            cv2.VideoWriter_fourcc(*'mp4v'),  # 备用MPEG-4编码[1](@ref)
            cv2.VideoWriter_fourcc(*'X264'),  # 开源H.264实现
            cv2.VideoWriter_fourcc(*'MJPG')  # 兼容性最好的编码器[2](@ref)
        ]

        # 确保帧率在合理范围内(1-60FPS)
        safe_fps = max(1, min(60, fps))
        print(f"安全帧率设置: {safe_fps}FPS")

        # 尝试不同编码器直到成功
        for fourcc in fourcc_options:
            try:
                self.out_video = cv2.VideoWriter(
                    self.current_video_path,
                    fourcc,
                    safe_fps,
                    (width, height)
                )

                # 检查是否成功初始化
                if self.out_video.isOpened():
                    print(f"✅ 成功使用编码器: {fourcc} 初始化视频写入器")
                    break
                else:
                    print(f"⚠️ 编码器 {fourcc} 初始化失败，尝试下一个选项")
                    self.out_video = None
            except Exception as e:
                print(f"编码器 {fourcc} 初始化错误: {str(e)}")
                self.out_video = None

        # 如果所有编码器都失败
        if self.out_video is None or not self.out_video.isOpened():
            print(f"❌ 错误: 无法初始化任何视频编码器，请检查系统支持")
            print("尝试使用默认编码器创建AVI文件")
            # 回退到AVI格式和XVID编码器[2](@ref)
            self.current_video_path = os.path.join(
                self.video_dir,
                f"segment_{self.segment_count}_{timestamp}.avi"
            )
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out_video = cv2.VideoWriter(
                self.current_video_path,
                fourcc,
                safe_fps,
                (width, height)
            )
            if not self.out_video.isOpened():
                print(f"❌ 严重错误: 无法初始化任何视频写入器")
                return False

        print(f"\n🔴 开始录制新视频片段: {self.current_video_path}")
        self.segment_start_time = time.time()
        self.segment_count += 1
        return True

    def predict_from_camera(self, show=True, camera_index=0):
        # 设置摄像头参数
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            raise ValueError(f"无法打开USB摄像头: /dev/video{camera_index}")

        print(f"成功打开USB摄像头: /dev/video{camera_index}")
        print("按 'q' 键退出实时检测...")

        # 获取摄像头参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 默认值
        self.actual_fps = fps  # 保存实际帧率

        print(f"摄像头参数: {width}x{height} @ {fps:.1f}FPS")

        # 初始化第一个视频写入器
        if not self.init_video_writer(width, height, fps):
            print("❌ 视频写入器初始化失败，程序终止")
            cap.release()
            return

        # 性能监控
        inference_times = []
        last_report_time = time.time()
        report_interval = 5  # 性能报告间隔(秒)
        last_frame_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法从摄像头获取帧")
                    time.sleep(0.1)
                    continue

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

                # 后处理（目标检测和标记）
                result_frame = self.postprocess(outputs, frame)

                # 帧处理总耗时
                frame_elapsed = (time.perf_counter() - frame_start) * 1000

                # 计算实际帧率
                current_time = time.time()
                actual_fps = 1.0 / (current_time - last_frame_time)
                last_frame_time = current_time

                # 检查是否需要创建新的视频片段
                if current_time - self.segment_start_time >= self.segment_interval:
                    # 关闭当前视频文件
                    if self.out_video is not None:
                        self.out_video.release()
                        print(f"🟢 视频片段已保存: {self.current_video_path}")
                        # 打印实际保存的视频帧数
                        segment_frame_count = int(self.segment_interval * self.actual_fps)
                        print(f"保存帧数: {segment_frame_count}, 实际FPS: {actual_fps:.1f}")

                    # 初始化新的视频写入器
                    self.init_video_writer(width, height, self.actual_fps)

                # 将标记后的帧写入当前视频文件
                if self.out_video is not None:
                    self.out_video.write(result_frame)

                # 显示处理后的帧
                if show:
                    # 在画面上显示实际FPS
                    fps_text = f"FPS: {actual_fps:.1f}"
                    cv2.putText(result_frame, fps_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('NPU加速-YOLO', result_frame)

                # 定时显示性能报告
                if current_time - last_report_time >= report_interval:
                    avg_time = np.mean(inference_times) if inference_times else 0
                    fps_estimate = 1000 / avg_time if avg_time > 0 else 0
                    print(f"\n📊 实时性能报告 (过去{report_interval}秒):")
                    print(f"- 处理帧数: {len(inference_times)}")
                    print(f"- 平均推理耗时: {avg_time:.2f}ms")
                    print(f"- 预估FPS: {fps_estimate:.1f}")
                    print(f"- 实际FPS: {actual_fps:.1f}")
                    print("-" * 40)

                    # 重置计数器和时间戳
                    inference_times = []
                    last_report_time = current_time

                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # 释放资源
            cap.release()
            if self.out_video is not None and self.out_video.isOpened():
                self.out_video.release()
                print(f"🟢 最后一个视频片段已保存: {self.current_video_path}")
            cv2.destroyAllWindows()

            # 最终性能总结报告
            if self.frame_count > 0:
                avg_time = self.total_inference_time / self.frame_count
                fps = 1000 / avg_time if avg_time > 0 else 0
                print("\n" + "=" * 50)
                print("最终性能分析报告:")
                print(f"- 总处理帧数: {self.frame_count}")
                print(f"- 平均推理耗时: {avg_time:.2f}ms")
                print(f"- 预估平均FPS: {fps:.1f}")
                print(f"- 保存视频片段数: {self.segment_count}")
                print(f"- 视频保存位置: {os.path.abspath(self.video_dir)}")
                print("=" * 50)


if __name__ == '__main__':
    inferencer = NPUInferencer(
        model_path="./runs/train/train6/weights/best.om",
        conf_threshold=0.3
    )
    inferencer.predict_from_camera(
        camera_index=0,
        show=False
    )