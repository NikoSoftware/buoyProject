import copy
import numpy as np
import cv2
import time
import acl
import acllite
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
from acllite_imageproc import AclLiteImageProc
import time


class ascend_yolo():
    def __init__(self,
                 model_path="./runs/train/train6/weights/best.om"):
        """
        初始化Yolo类
        Args:
            model_path: om模型的路径
            class_list: 类别列表，默认为["0"]
        """

        # 加载模型
        self.model = AclLiteModel(model_path)

        # 设置类别列表和标准输入尺寸
        class_list = ["buoy"]
        self.class_list = class_list if class_list else ["0"]
        self.std_h, self.std_w = 640, 640

    def resize_image(self, image, size, letterbox_image):
        """
        对输入图像进行resize
        Args:
            image: 输入图像
            size: 目标尺寸
            letterbox_image: bool 是否进行letterbox变换
        Returns: 指定尺寸的图像
        """
        ih, iw, _ = image.shape
        h, w = size
        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            # 生成画布
            image_back = np.ones((h, w, 3), dtype=np.uint8) * 114
            # 将image放在画布中心区域-letterbox
            image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
        else:
            image_back = image
        return image_back

    def img2input(self, img):
        """
        将图像转换为模型输入格式
        """
        return np.expand_dims(img, axis=0).astype(np.uint8)

    def std_output(self, pred):
        """
        将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
        """
        pred = np.squeeze(pred)
        pred = np.transpose(pred, (1, 0))
        pred_class = pred[..., 4:]
        pred_conf = np.max(pred_class, axis=-1)
        pred = np.insert(pred, 4, pred_conf, axis=-1)
        return pred

    def xywh2xyxy(self, *box):
        """
        将xywh转换为左上角点和左下角点
        """
        ret = [box[0] - box[2] // 2, box[1] - box[3] // 2,
               box[0] + box[2] // 2, box[1] + box[3] // 2]
        return ret

    def get_inter(self, box1, box2):
        """
        计算相交部分面积
        """
        x1, y1, x2, y2 = self.xywh2xyxy(*box1)
        x3, y3, x4, y4 = self.xywh2xyxy(*box2)
        # 验证是否存在交集
        if x1 >= x4 or x2 <= x3:
            return 0
        if y1 >= y4 or y2 <= y3:
            return 0
        # 将x1,x2,x3,x4排序，因为已经验证了两个框相交，所以x3-x2就是交集的宽
        x_list = sorted([x1, x2, x3, x4])
        x_inter = x_list[2] - x_list[1]
        # 将y1,y2,y3,y4排序，因为已经验证了两个框相交，所以y3-y2就是交集的宽
        y_list = sorted([y1, y2, y3, y4])
        y_inter = y_list[2] - y_list[1]
        # 计算交集的面积
        inter = x_inter * y_inter
        return inter

    def get_iou(self, box1, box2):
        """
        计算交并比
        """
        box1_area = box1[2] * box1[3]  # 计算第一个框的面积
        box2_area = box2[2] * box2[3]  # 计算第二个框的面积
        inter_area = self.get_inter(box1, box2)
        union = box1_area + box2_area - inter_area  # (A n B)/(A + B - A n B)
        iou = inter_area / union
        return iou

    def nms(self, pred, conf_thres, iou_thres):
        """
        非极大值抑制nms
        """
        box = pred[pred[..., 4] > conf_thres]  # 置信度筛选
        cls_conf = box[..., 5:]
        cls = []
        for i in range(len(cls_conf)):
            cls.append(int(np.argmax(cls_conf[i])))
        total_cls = list(set(cls))  # 记录图像内共出现几种物体
        output_box = []
        # 每个预测类别分开考虑
        for i in range(len(total_cls)):
            clss = total_cls[i]
            cls_box = []
            temp = box[:, :6]
            for j in range(len(cls)):
                # 记录[x,y,w,h,conf(最大类别概率),class]值
                if cls[j] == clss:
                    temp[j][5] = clss
                    cls_box.append(temp[j][:6])
            #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
            cls_box = np.array(cls_box)
            sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
            # 得到置信度最大的预测框
            max_conf_box = sort_cls_box[0]
            output_box.append(max_conf_box)
            sort_cls_box = np.delete(sort_cls_box, 0, 0)
            # 对除max_conf_box外其他的框进行非极大值抑制
            while len(sort_cls_box) > 0:
                # 得到当前最大的框
                max_conf_box = output_box[-1]
                del_index = []
                for j in range(len(sort_cls_box)):
                    current_box = sort_cls_box[j]
                    iou = self.get_iou(max_conf_box, current_box)
                    if iou > iou_thres:
                        # 筛选出与当前最大框Iou大于阈值的框的索引
                        del_index.append(j)
                # 删除这些索引
                sort_cls_box = np.delete(sort_cls_box, del_index, 0)
                if len(sort_cls_box) > 0:
                    output_box.append(sort_cls_box[0])
                    sort_cls_box = np.delete(sort_cls_box, 0, 0)
        return output_box

    def cod_trf(self, result, pre, after):
        """
        将预测框的坐标映射回原图像上
        """
        res = np.array(result)
        if not result or len(result) == 0:
            return np.array([])
        x, y, w, h, conf, cls = res.transpose((1, 0))
        x1, y1, x2, y2 = self.xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
        h_pre, w_pre, _ = pre.shape
        h_after, w_after, _ = after.shape
        scale = max(w_pre / w_after, h_pre / h_after)  # 缩放比例
        h_pre, w_pre = h_pre / scale, w_pre / scale  # 计算原图在等比例缩放后的尺寸
        x_move, y_move = abs(w_pre - w_after) // 2, abs(h_pre - h_after) // 2  # 计算平移的量
        ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
        ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
        ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
        return ret

    def draw(self, res, image, cls=None):
        """
        将预测框绘制在image上
        """
        if cls is None:
            cls = self.class_list

        for r in res:
            # 画框
            image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 0), 1)
            # 表明类别
            text = "{}:{}".format(cls[int(r[5])], round(float(r[4]), 2))
            h, w = int(r[3]) - int(r[1]), int(r[2]) - int(r[0])  # 计算预测框的长宽
            font_size = min(h / 640, w / 640) * 3  # 计算字体大小（随框大小调整）
            image = cv2.putText(image, text, (max(10, int(r[0])), max(20, int(r[1]))),
                                cv2.FONT_HERSHEY_COMPLEX, max(font_size, 0.3), (0, 0, 255), 1)
        return image

    def detect(self, image, conf_thres=0.7, iou_thres=0.4, draw_result=True):
        """
        检测图像中的手
        Args:
            image: 输入图像
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            draw_result: 是否在图像上绘制结果
        Returns:
            result: 检测结果 [x1, y1, x2, y2, conf, class]
            output_image: 绘制了检测框的图像（如果draw_result=True）
        """
        # 复制输入图像，避免修改原始图像
        inferimg = copy.deepcopy(image)

        # 前处理
        img_after = self.resize_image(inferimg, (self.std_w, self.std_h), True)
        data = self.img2input(img_after)

        # 推理
        pred = self.model.execute([data])[0]

        # 后处理
        pred = self.std_output(pred)
        result = self.nms(pred, conf_thres, iou_thres)
        result = self.cod_trf(result, inferimg, img_after)

        if draw_result:
            output_image = self.draw(result, inferimg)
            return result, output_image
        else:
            return result, image

    def benchmark(self, images, count=1000):
        """
        性能测试函数
        Args:
            image_path: 测试图像的路径
            count: 测试次数
        Returns:
            性能数据
        """
        init_img = images.copy()
        if init_img.size == 0:
            print("路径有误！")
            return

        all_time = time.time()
        pre_time = 0
        infer_time = 0
        post_time = 0

        for i in range(count):
            # 前处理性能
            start_time = time.time()
            inferimg = copy.deepcopy(init_img)
            img_after = self.resize_image(inferimg, (self.std_w, self.std_h), True)
            data = self.img2input(img_after)
            pre_time += time.time() - start_time

            # 推理性能
            start_time = time.time()
            pred = self.model.execute([data])[0]
            infer_time += time.time() - start_time

            # 后处理性能
            start_time = time.time()
            pred = self.std_output(pred)
            result = self.nms(pred, 0.7, 0.4)
            result = self.cod_trf(result, inferimg, img_after)
            image = self.draw(result, inferimg)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            post_time += time.time() - start_time

        total_time = time.time() - all_time
        fps = count / total_time

        return {
            "pre_time_avg": pre_time / count,
            "infer_time_avg": infer_time / count,
            "post_time_avg": post_time / count,
            "total_time": total_time,
            "fps": fps
        }


if __name__ == '__main__':
    import acl
    import acllite
    from acllite_model import AclLiteModel
    from acllite_resource import AclLiteResource
    from acllite_imageproc import AclLiteImageProc
    import time

    acl_resource = AclLiteResource()
    acl_resource.init()
    # 初始化YoloHand
    yolo = ascend_yolo()

    # 单次检测示例
    img_path = "./datasets/test/20250611213828.png"
    image = cv2.imread(img_path)

    # benchmark_result=yolo.benchmark(image)
    # print(benchmark_result)
    # # 检测并绘制结果
