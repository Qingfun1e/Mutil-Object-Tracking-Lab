import cv2
from ultralytics import YOLO

class BBoxes:
    def __init__(self, boxes, cls, conf):
        # 保持为 tensor 形式
        self.xyxy = boxes          # tensor (N,4)
        self.cls = cls             # tensor (N,)
        self.conf = conf           # tensor (N,)

class dectector:
    def __init__(self, model_name='../models/yolov8n.pt', conf_threshold=0.2,device='cuda'):
        self.model = YOLO(model_name)
        self.model.to(device)
        self.labels = self.model.names
        self.conf_threshold = conf_threshold

    def detect(self, image):
        results = self.model.predict(image, verbose=False,classes=range(8))[0]  # 单帧结果
        boxes = results.boxes.xyxy # tensor (N,4)
        cls = results.boxes.cls    # tensor (N,)
        conf = results.boxes.conf  # tensor (N,)

        # 过滤掉置信度小于阈值的框
        mask = conf >= self.conf_threshold
        boxes = boxes[mask]
        cls = cls[mask]
        conf = conf[mask]

        return BBoxes(boxes, cls, conf)


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(r'../videos/test_videos.mp4')
    model_name = '../models/yolov8n.pt'
    model = YOLO(model_name)
    model.to('cuda')
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = model.predict(frame, show=True)
