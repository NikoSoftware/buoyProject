import cv2
import numpy as np
from tqdm import tqdm
from ais_bench.infer.interface import InferSession  # 昇腾推理接口


class NPUInferencer:
    def __init__(self, model_path, conf_threshold=0.3):
        """
        初始化NPU推理引擎
        :param model_path: .om模型路径
        :param conf_threshold: 置信度阈值
        """
        # 加载NPU模型[3,7](@ref)
        self.model = InferSession(device_id=0, model_path=model_path)
        self.conf_threshold = conf_threshold
        # 获取输入输出信息[9](@ref)
        self.input_shape = self.model.get_inputs()[0].shape  # [1,3,640,640]
        self.output_shape = self.model.get_outputs()[0].shape  # [1,84,8400]

    def preprocess(self, frame):
        """
        图像预处理（适配NPU输入）
        :param frame: OpenCV图像帧 (HWC格式)
        :return: 预处理后的张量(NCHW)
        """
        # 调整尺寸并转换颜色空间[9](@ref)
        img = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
        img = img.transpose(2, 0, 1)[np.newaxis]  # HWC→NCHW
        img = img.astype("float32") / 255.0  # 归一化
        return img

    def postprocess(self, outputs, frame):
        """
        后处理：解析NPU输出并绘制结果
        :param outputs: NPU输出数据
        :param frame: 原始图像帧
        """
        # YOLOv8输出格式: [1,84,8400] (4坐标+80类别)[7](@ref)
        detections = outputs[0][0].transpose(1, 0)  # 转置为[8400,84]

        # 过滤低置信度框
        conf_mask = detections[:, 4] > self.conf_threshold
        detections = detections[conf_mask]

        # 绘制检测框
        for *xyxy, conf, cls_id in detections:
            x1, y1, x2, y2 = map(int, xyxy[:4])
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            label = f"Class {int(cls_id)} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def predict(self, source, show=True):
        """
        NPU加速视频推理
        :param source: 视频路径
        :param show: 是否实时显示
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")

        # 获取视频参数[3](@ref)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 性能优化配置[3](@ref)
        import os
        os.environ['TE_PARALLEL_COMPILER'] = '1'  # 限制编译线程防止OOM

        # 逐帧推理
        for _ in tqdm(range(total_frames), desc="NPU推理进度"):
            ret, frame = cap.read()
            if not ret: break

            # NPU推理流程
            blob = self.preprocess(frame)
            outputs = self.model.infer(blob, mode="static")  # 静态模式优化延迟[7](@ref)

            # 后处理与显示
            result_frame = self.postprocess(outputs, frame)
            if show:
                cv2.imshow('NPU加速-YOLO', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 使用示例（需提前转换OM模型）
    inferencer = NPUInferencer(
        model_path="./runs/train/train6/weights/best.om",  # 替换为转换后的OM模型
        conf_threshold=0.3
    )
    inferencer.predict(
        source=r'./datasets/test/30386095338-1-192.mp4',
        show=True
    )