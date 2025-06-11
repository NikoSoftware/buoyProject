

from ultralytics import YOLO

import cv2




def run():
    # source = r'./datasets/test/20250611213828.png'
    source = r'./datasets/test/30386095338-1-192.mp4'
    model = YOLO("./runs/train/train6/weights/best.pt")
    model.predict(source,conf=0.3,show=True)
    cv2.waitKey()






if __name__ == '__main__':
    run()