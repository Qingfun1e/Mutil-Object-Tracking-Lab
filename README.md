# 基于IoU贪心匹配的多目标追踪视频播放器

简单描述各个功能的设计思路，系统测试见演示视频，代码见github链接


# 1. 组件设计
## 1.1 帧捕捉模块
采用opencv的VideoCapture类创建帧捕捉器对象，视频源支持在线摄像头/IP摄像头或离线视频文件。通过捕捉器的read函数从cv2.mat矩阵中获取一帧图像。

## 1.2 目标检测模块
通过ultralytics库快速调用YOLO模型，选用性能与效率兼顾的yolov8s模型。检测器对帧捕捉模块获取的视频帧执行检测，得到当前帧下的所有边界框（bounding box），并通过置信度阈值筛选掉低置信度框。每个帧检测出的边界框包含cls（类别）、conf（置信度）、xyxy（坐标）三个tensor。

## 1.3 追踪器模块
基于IoU贪心匹配实现多目标追踪，参考论文《High-Speed Tracking-by-Detection Without Using Image Information》的算法。  
每个轨迹（track）以字典形式存储，包含ID、cls、boxes(xyxy)_list、conf_list、life_time、hit_cnt等键值对。算法维护两个列表：active_track（活跃轨迹）和finished_track（已结束轨迹）。  
具体逻辑：每帧检测得到的边界框与上一帧active_track中各轨迹的最新边界框（boxes_list最后一个元素）计算IoU矩阵；通过torch.max找到每个轨迹在当前帧的最佳匹配框，若IoU大于设定阈值，则将该框作为轨迹延续并从当前帧边界框中移除，否则轨迹的life_time减1，当life_time为0时销毁轨迹，若销毁轨迹的hit_cnt大于min_hit则加入finished_track；未匹配的边界框创建为新轨迹并加入active_track，循环至所有视频帧处理完成。

## 1.4 可视化模块
对视频帧及当前帧追踪器中的active_track进行可视化：当轨迹的hit_cnt大于min_hit时，绘制其检测框及ID、CLS、CONF信息，同时显示FPS、播放速度及按键交互提示等内容。


# 2. 多线程设计
## 2.1 核心思想
多目标追踪任务采用“生产者-消费者”模型：视频帧捕捉线程为生产者，“检测-追踪-绘制”线程为消费者。通过queue实现线程通信（存储视频帧），利用python queue库自带的安全锁机制避免资源互斥，无需额外加锁。

## 2.2 相关细节
- 生产者线程：持续将捕捉的帧送入frame_queue，队列满时自动丢弃旧帧，无需等待消费者，确保捕捉效率。
- 消费者线程：整合“检测-跟踪-可视化”模块，因可视化时延较短，未拆分额外队列（测试显示拆分队列会导致FPS下降5%）。

## 2.3 多线程控制
- 线程启停：通过running标志控制线程启动，主循环检测到running为false时停止所有线程。
- 有序关停：生产者线程完成捕捉后将capture_finished标志置1，消费者线程若无法从frame_queue获取帧且capture_finished为1，则自动关闭。
- 启动优化：因YOLO检测器需编译CUDA kernel，使用warm_up字段对消费者线程预推理，预热完成后再正常运行，避免生产者线程启动后丢帧。


# 3. 交互设计
## 3.1 核心思想
基于cv2.imshow实现可视化，按键交互逻辑集成于消费者线程：通过pause字段控制暂停功能（生产者与消费者线程均判断该字段），不影响可视化与交互；通过cv2.waitKey捕捉按键事件。

## 3.2 相关细节
- 暂停功能：检测到空格键时，将pause字段置反（初始为false），使生产者与消费者线程跳过对应代码块（停止捕捉帧与“检测-追踪-绘制”）。
- 单步功能：在pause为true时生效，按下D键将step_forward字段置1，处理一帧后复位，实现单帧步进。
- 重置ID：直接重置tracker的active_track列表和track_ID变量。
- 播放速度控制：通过动态调整生产者线程中帧捕捉的时间间隔（匹配当前FPS）实现，因帧捕捉I/O操作时间固定，间隔调整可直接控制播放速度。
- FPS计算：使用deque存储近三帧的处理时间，取平均值（单帧时间从消费者线程获取帧开始，至绘制结束为止）。