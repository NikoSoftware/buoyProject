

from ultralytics import YOLO

import cv2




def run():
    # source = r'./datasets/test/20250611213828.png'
    source = r'./datasets/test/'
    model = YOLO("./runs/train/train2/weights/best.pt")
    model.predict(source,conf=0.4,show=True,save_txt=True)
    cv2.waitKey()






if __name__ == '__main__':
    run()
