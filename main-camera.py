import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession


class NPUInferencer:
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = InferSession(device_id=0, model_path=model_path)
        self.conf_threshold = conf_threshold
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_shape = self.model.get_outputs()[0].shape
        self.total_inference_time = 0
        self.frame_count = 0

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

        if outputs[0].shape == (1, 8400, 84):
            detections = outputs[0][0]
        else:
            detections = outputs[0][0].transpose(1, 0)

        if detections.size == 0:
            return frame

        for row in detections:
            if len(row) < 6: continue
            try:
                x1, y1, x2, y2, conf, cls_id = row[:6]
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                if conf < self.conf_threshold: continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except ValueError:
                continue

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"后处理耗时: {elapsed:.2f}ms")
        return frame

    def predict_from_camera(self, show=True, camera_index=0):
        # 设置摄像头参数
        cap = cv2.VideoCapture(camera_index)  # 使用默认USB摄像头
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置分辨率[3,6](@ref)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率[3,6](@ref)

        if not cap.isOpened():
            # 尝试解决摄像头权限问题[1](@ref)
            import os
            os.system(f"sudo chmod 666 /dev/video{camera_index}")
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise ValueError(f"无法打开USB摄像头: /dev/video{camera_index}")

        print(f"成功打开USB摄像头: /dev/video{camera_index}")
        print("按 'q' 键退出实时检测...")

        # 性能监控数据结构
        inference_times = []
        last_report_time = time.time()
        report_interval = 5  # 性能报告间隔(秒)

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

                # 后处理
                result_frame = self.postprocess(outputs, frame)

                # 帧处理总耗时
                frame_elapsed = (time.perf_counter() - frame_start) * 1000

                # 显示处理后的帧
                if show:
                    cv2.imshow('NPU加速-YOLO', result_frame)

                # 定时显示性能报告
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    avg_time = np.mean(inference_times) if inference_times else 0
                    fps = 1000 / avg_time if avg_time > 0 else 0
                    print(f"\n实时性能报告 (过去{report_interval}秒):")
                    print(f"- 处理帧数: {self.frame_count}")
                    print(f"- 平均推理耗时: {avg_time:.2f}ms")
                    print(f"- 预估FPS: {fps:.1f}")
                    print("-" * 40)

                    # 重置计数器和时间戳
                    inference_times = []
                    last_report_time = current_time

                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # 最终性能总结报告
            if self.frame_count > 0:
                avg_time = self.total_inference_time / self.frame_count
                fps = 1000 / avg_time
                print("\n" + "=" * 50)
                print("最终性能分析报告:")
                print(f"- 总处理帧数: {self.frame_count}")
                print(f"- 平均推理耗时: {avg_time:.2f}ms")
                print(f"- 预估平均FPS: {fps:.1f}")
                print("=" * 50)


if __name__ == '__main__':
    inferencer = NPUInferencer(
        model_path="./runs/train/train6/weights/best.om",
        conf_threshold=0.3
    )
    inferencer.predict_from_camera(
        camera_index=0,  # 使用第一个USB摄像头
        show=False
    )