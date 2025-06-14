import cv2
import numpy as np
import time  # 新增时间模块
from tqdm import tqdm
from ais_bench.infer.interface import InferSession


class NPUInferencer:
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = InferSession(device_id=0, model_path=model_path)
        self.conf_threshold = conf_threshold
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_shape = self.model.get_outputs()[0].shape
        self.total_inference_time = 0  # 新增：累计推理耗时
        self.frame_count = 0  # 新增：帧计数器

    def preprocess(self, frame):
        start_time = time.perf_counter()  # 记录预处理开始时间[1,2](@ref)
        img = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)[np.newaxis]
        img = img.astype("float32") / 255.0
        elapsed = (time.perf_counter() - start_time) * 1000  # 毫秒计算[1](@ref)
        print(f"预处理耗时: {elapsed:.2f}ms")
        return img

    def postprocess(self, outputs, frame):
        start_time = time.perf_counter()  # 记录后处理开始时间[1](@ref)
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

    def predict(self, source, show=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"视频总帧数: {total_frames}, 原始FPS: {fps:.1f}")

        # 性能监控数据结构
        inference_times = []

        for _ in tqdm(range(total_frames), desc="NPU推理进度"):
            ret, frame = cap.read()
            if not ret: break

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

            # 后处理
            result_frame = self.postprocess(outputs, frame)

            # 帧处理总耗时
            frame_elapsed = (time.perf_counter() - frame_start) * 1000
            print(f"单帧总耗时: {frame_elapsed:.2f}ms")

            if show:
                cv2.imshow('NPU加速-YOLO', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
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
            print("=" * 50)


if __name__ == '__main__':
    inferencer = NPUInferencer(
        model_path="./runs/train/train6/weights/best.om",
        conf_threshold=0.3
    )
    inferencer.predict(
        source=r'./datasets/test/30386095338-1-192.mp4',
        show=False
    )