import cv2

class VideoCapture:
    '''
    cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)
    是OpenCV在Windows平台下用于指定使用DirectShow作为视频捕获后端的一种方式。
    这里的 cv2.CAP_DSHOW 是一个常量，表示使用DirectShow（DirectX Show）技术来访问和控制摄像头。
    cv2.CAP_DSHOW 参数含义：
    当你将 cv2.CAP_DSHOW 与摄像头编号相加时，
    实际上是告诉OpenCV使用DirectShow框架来打开并操作指定的摄像头设备。
    DirectShow是Windows操作系统中用于处理音/视频流的一种API集合，
    它可以动态构建数据源到渲染器之间的过滤器链，支持多种音频和视频格式，并具有一定的硬件加速能力。
    '摘自https://www.cnblogs.com/zililove/p/18076353'
    '''
    def __init__(self, video_source_type='camera', video_source=None):
        '''
        :param video_source_type:camera or video 视频源类型
        :param video_source: 视频源
        :param FPS: 视频每秒帧数
        :param frame_queue_size:视频队列尺寸
        '''
        # judge the source type
        if video_source_type == 'camera':
            self.camera_number = video_source
            self.video_capture = cv2.VideoCapture(self.camera_number+cv2.CAP_DSHOW)
            if not self.video_capture.isOpened():#TODO 改GUI弹窗
                raise ValueError("Unable to open camera source", self.camera_number)
        elif video_source_type == 'video':
            self.video_source = video_source
            self.video_capture = cv2.VideoCapture(self.video_source)
            if not self.video_capture.isOpened():
                raise ValueError("Unable to open video source", self.video_source)
    #     self.frame_queue = queue.Queue(frame_queue_size)
    #     self.module_running = False
    #     self.fps = FPS
    #     self.thread = None # pre-defined
    #     self.lock = threading.Lock()
    #
    # def start(self):
    #     # start the threading module
    #     self.module_running = True
    #     self.thread = threading.Thread(target=self.read_frame, daemon=True)
    #     self.thread.start()
    #
    # def get_frame(self):
    #     # get frame
    #     try:
    #         return self.frame_queue.get(block=False)
    #     except queue.Empty:
    #         return None
    def read_frame(self):
        ret, frame = self.video_capture.read()
        return ret, frame
        # # a treading function for reading frame which is pushed into the queue
        # while self.module_running:
        #     ret, frame = self.video_capture.read()
        #     # self.video_capture.read() returns a tuple (bool, frame)
        #     if self.frame_queue.full():
        #         self.frame_queue.get()
        #     # when the queue is full, discard the oldest frame
        #     if ret:
        #         with self.lock:
        #             self.frame_queue.put(frame)
        #     else:
        #         break
        #     # control the frame rate
        #     time.sleep(1 / self.fps)

if __name__ == '__main__':
    #test
    video_capture = VideoCapture(video_source_type='video', video_source=r'../videos/test_videos.mp4', FPS=60, frame_queue_size=10)

    video_capture.start()
    while True:
        frame = video_capture.get_frame()
        if frame is not None:
            cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.stop()
    cv2.destroyAllWindows()