import cv2
import time
import signal
import sys

width = 640
height = 640

def crop_frame(frame):

    rotate = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    h, w = rotate.shape[:2]

    start_y = (h - 640)//2
    start_x = (w - 640)//2

    crop = rotate[start_y:start_y+640, start_x:start_x+640]

    return crop



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # def signal_handler(sig, frame):
    #     print("\n捕获到Ctrl+Z信号，正在释放摄像头资源...")
    #     cap.release()  # 释放摄像头
    #     cv2.destroyAllWindows()  # 关闭窗口
    #     sys.exit(0)  # 优雅退出
    #
    # signal.signal(signal.SIGTSTP, signal_handler)


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("USB 摄像头没有打开")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = crop_frame(frame)

        imgName = './images/frame%d.jpg' % int(time.time())

        cv2.imwrite(imgName, frame)

        print(imgName)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:  # ESC退出
            break

        time.sleep(0.5)


    cap.release()
    cv2.destroyAllWindows()
