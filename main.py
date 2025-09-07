import cv2
import torch
import time
import threading
import queue
import logging
import yaml
from collections import deque
import os

from video_capture import capture
from object_detection import yolo_detector
from object_tracking import tracking_by_IoUmatching
from visualization import draw_bbox


# ======================
# 读取配置
# ======================
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config("config.yaml")

# ======================
# 初始化日志
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ======================
# 解析配置
# ======================
MODEL_NAME = config["model"]["name"]
CONF_THRESHOLD = config["model"]["conf_threshold"]
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
    if config["model"]["device"] == "auto"
    else config["model"]["device"]
)

MAX_AGE = config["tracker"]["max_age"]
MIN_HITS = config["tracker"]["min_hits"]
IOU_THRESHOLD = config["tracker"]["iou_threshold"]

VIDEO_TYPE = config["video"]["type"]
VIDEO_SOURCE = config["video"]["source"]

FRAME_QUEUE_SIZE = config["system"]["frame_queue_size"]
DETECTION_QUEUE_SIZE = config["system"]["detection_queue_size"]
CAP_FPS = config["system"]["cap_fps"]
PLAY_SPEED = config.get("playback", {}).get("initial_speed", 1.0)  # 获取初始播放速度
# ======================
# 初始化模型与视频源
# ======================
detector = yolo_detector.dectector(
    model_name=MODEL_NAME,
    conf_threshold=CONF_THRESHOLD,
    device=DEVICE
)
cls_list = detector.labels

tracker = tracking_by_IoUmatching.IoUTracker(
    max_age=MAX_AGE,
    min_hits=MIN_HITS,
    iou_threshold=IOU_THRESHOLD
)

video_capture = capture.VideoCapture(
    video_source_type=VIDEO_TYPE,
    video_source=VIDEO_SOURCE,
)


class TwoThreadMOT:
    def __init__(self, video_capture, detector, tracker, cls_list):
        self.video_capture = video_capture
        self.detector = detector
        self.tracker = tracker
        self.cls_list = cls_list

        # FPS 相关
        self.frame_times = deque(maxlen=60)  # 最近60帧耗时
        self.smooth_fps = 0.0

        self.play_speed = PLAY_SPEED  # 播放速度
        self.running = False  # 用于线程启动
        self.model_ready = False  # 模型是否加载完
        self.capture_finished = False  # 判断帧捕捉是否完成
        self.pause = False # 暂停标志s
        self.step_forward = False  # 单步播放标志
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        
        # 轨迹绘制器
        self.trajectory_drawer = None
        
        # 截图相关
        self.screenshot_count = 0
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

        # 性能分析相关
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_update_interval = 0.5  # 每0.5秒更新一次FPS显示

    def warmup_model(self):
        """加载模型并进行一次预热"""
        logging.info("正在加载模型并进行预热...")
        ret, frame = self.video_capture.read_frame()
        if ret:
            _ = self.detector.detect(frame)  # 只跑一次，不关心结果
            logging.info("模型预热完成")
            self.model_ready = True
        else:
            logging.warning("视频源读取失败，无法预热模型")
            self.model_ready = True  # 仍然置为True，避免卡死

    def start(self):
        # 先预热模型
        self.warmup_model()

        self.running = True
        self.capture_t = threading.Thread(target=self.capture_thread, daemon=True)
        self.track_t = threading.Thread(target=self.tracking_thread, daemon=True)
        self.capture_t.start()
        self.track_t.start()
        logging.info("双线程MOT系统已启动")

    def stop(self):
        self.running = False
        logging.info("正在停止系统...")

        if self.capture_t.is_alive():
            self.capture_t.join()
        if self.track_t.is_alive():
            self.track_t.join()

        self.video_capture.video_capture.release()
        cv2.destroyAllWindows()
        logging.info("系统已安全退出")

    def capture_thread(self):
        try:
            while self.running:
                if not self.model_ready:
                    time.sleep(0.01)
                    continue
                if not self.pause or self.step_forward:
                    ret, frame = self.video_capture.read_frame()
                    if not ret:
                        logging.info("视频已结束，采集线程退出")
                        self.capture_finished = True
                        break

                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get(False)  # 非阻塞移除最旧帧
                            logging.debug("帧队列已满，已丢弃一帧")
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                    if self.step_forward:
                        self.step_forward = False  # 处理完一帧后重置单步标志
                time.sleep(1 / (CAP_FPS * self.play_speed)) if CAP_FPS > 0 else None
        except Exception as e:
            logging.error(f"捕获线程异常: {e}")

    def tracking_thread(self):
        try:
            while self.running:
                if not self.pause or self.step_forward:
                    if self.frame_queue.empty():
                        if self.capture_finished:  # 捕获线程已结束且没帧了
                            logging.info("帧队列已空，跟踪线程退出")
                            self.running = False  # 只改标志，不调用 stop()
                            break
                        else:
                            time.sleep(0.001)  # 短暂休眠避免忙等待
                            continue

                    # 使用非阻塞方式获取帧
                    try:
                        frame = self.frame_queue.get(False)
                    except queue.Empty:
                        continue

                    # 初始化轨迹绘制器（在第一次获取帧时）
                    if self.trajectory_drawer is None:
                        self.trajectory_drawer = draw_bbox.TrajectoryDrawer(frame.shape)

                    # 性能计时开始
                    frame_start_time = time.time()

                    # 目标检测
                    detect_start_time = time.time()
                    bboxes = self.detector.detect(frame)
                    detect_time = time.time() - detect_start_time

                    # 目标跟踪
                    track_start_time = time.time()
                    self.tracker.update(bboxes)
                    track_time = time.time() - track_start_time

                    # 可视化绘制（传入平滑 FPS、暂停状态和播放速度）
                    draw_start_time = time.time()
                    frame = self.trajectory_drawer.draw(self.tracker, frame, MIN_HITS, self.cls_list, 
                                                       fps=self.smooth_fps, is_paused=self.pause, 
                                                       play_speed=self.play_speed)
                    draw_time = time.time() - draw_start_time

                    # 计算 FPS
                    frame_end_time = time.time()
                    self.frame_times.append(frame_end_time - frame_start_time)
                    if len(self.frame_times) > 1:
                        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                        self.smooth_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

                    # 每秒更新一次日志
                    if int(frame_end_time) != int(frame_start_time):
                        logging.info(
                            f"检测: {detect_time * 1000:.2f}ms, "
                            f"跟踪: {track_time * 1000:.2f}ms, "
                            f"绘制: {draw_time * 1000:.2f}ms, "
                            f"总计: {(frame_end_time - frame_start_time) * 1000:.2f}ms, "
                            f"FPS: {self.smooth_fps:.2f}"
                        )

                # 窗口显示和交互
                cv2.imshow('Tracking', frame)
                key = cv2.waitKey(1)
                if key == (ord('q') or 27):
                    self.running = False  # 只改标志
                    break
                if key == ord(' '):
                    self.pause = not self.pause
                    logging.info(f"系统已{'暂停' if self.pause else '恢复'}")
                if key == 83 or key == ord('d'):  # Right arrow 或 'd' 单步
                    if self.pause:
                        self.step_forward = True
                        logging.info("单步播放下一帧")
                if key == ord('r'):
                    self.tracker.reset()
                    logging.info("已重置跟踪器")
                # 播放速度控制
                if key == ord('+') or key == ord('='):
                    self.play_speed = min(self.play_speed * 2, 4)  # 最大4倍速
                    logging.info(f"播放速度: {self.play_speed}x")
                if key == ord('-') or key == ord('_'):
                    self.play_speed = max(self.play_speed / 2, 0.25)  # 最小0.25倍速
                    logging.info(f"播放速度: {self.play_speed}x")
                if key == ord('0') or key == ord(')'):  # 重置速度
                    self.play_speed = PLAY_SPEED
                    logging.info(f"播放速度已重置为: {self.play_speed}x")
                # 截图功能
                if key == ord('c') or key == ord('C'):
                    self.screenshot_count += 1
                    screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{self.screenshot_count:04d}.png")
                    cv2.imwrite(screenshot_path, frame)
                    logging.info(f"截图已保存: {screenshot_path}")

        except Exception as e:
            logging.error(f"跟踪线程异常: {e}")


if __name__ == "__main__":
    mot_system = TwoThreadMOT(video_capture, detector, tracker, cls_list)
    mot_system.start()
    logging.info("系统启动完成，按 'q' 或 ESC 退出，按空格键暂停/恢复，暂停时按右箭头键或'd'键单步播放")
    logging.info("按 '+' 或 '=' 加速播放，按 '-' 或 '_' 减慢播放，按 '0' 重置播放速度，按 'C' 截图")
    try:
        while mot_system.running:   # 主线程循环
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        mot_system.stop()