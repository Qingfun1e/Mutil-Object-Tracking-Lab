import cv2
from ultralytics import YOLO
import time


def simple_video_detection():
    # 初始化模型
    model = YOLO('models/yolov8m.pt').to('cuda')

    # 使用YOLO内置的视频处理功能
    results = model.track(
        source='videos/test_videos2.mp4',
        stream=True,
        conf=0.25,
        show=True,  # 直接显示结果
        verbose=False,
        device='cuda'
    )

    # 只需迭代结果即可
    for result in results:
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    simple_video_detection()