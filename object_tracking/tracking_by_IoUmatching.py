import torch
class IoUTracker:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3, device='cpu'):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.device = device
        #track:{ID, Boxes, confs, cls, hit_cnt,lifetime}
        self.active_tracks = []
        # self.finished_tracks = []
        self.track_id = 0

    def update(self, bboxes):
        '''
        :param bboxes: detector's bboxes
        :return:
        '''
        # 如果没有检测框，直接处理轨迹生命周期
        if len(bboxes.xyxy) == 0:
            new_tracks = []
            for track in self.active_tracks:
                track['lifetime'] -= 1
                if track['lifetime'] <= 0 and track['hit_cnt'] >= self.min_hits:
                    # 轨迹结束条件
                    pass  # 可以添加轨迹完成处理
                elif track['lifetime'] > 0:
                    new_tracks.append(track)
            self.active_tracks = new_tracks
            return

        # 只有在张量不在正确设备上时才进行迁移
        if bboxes.xyxy.device != torch.device(self.device):
            bboxes.xyxy = bboxes.xyxy.to(self.device)
        if bboxes.cls.device != torch.device(self.device):
            bboxes.cls = bboxes.cls.to(self.device)
        if bboxes.conf.device != torch.device(self.device):
            bboxes.conf = bboxes.conf.to(self.device)
        
        # detections:bbox
        cls = bboxes.cls[:, None]
        confs = bboxes.conf[:, None]
        boxes = bboxes.xyxy
        detections = torch.cat((boxes, cls, confs), dim=1)
        #(x1, y1, x2, y2, cls, conf)
        new_tracks = []
        
        # 如果没有活跃轨迹但有检测框，直接创建新轨迹
        if len(self.active_tracks) == 0:
            for i, det in enumerate(detections):
                new_track = {
                    'id': self.track_id,
                    'boxes': [det[0:4]],
                    'cls': int(det[4]),
                    'confs': [det[5].item()],
                    'hit_cnt': 1,
                    'lifetime': self.max_age
                }
                new_tracks.append(new_track)
                self.track_id += 1
            self.active_tracks = new_tracks
            return

        # 正常跟踪处理
        # 预先将所有轨迹框放到一个张量中以提高效率
        track_boxes = torch.stack([track['boxes'][-1] for track in self.active_tracks]).to(self.device)
        
        # 计算所有轨迹和检测之间的IoU
        iou_matrix = calculate_iou_matrix(track_boxes, detections[:, 0:4], self.device)
        
        # 为每个轨迹找到最佳匹配
        track_matches = torch.max(iou_matrix, dim=1)  # (values, indices)
        matched_detections = [False] * len(detections)  # 使用列表代替集合以提高性能
        
        for i, track in enumerate(self.active_tracks):
            if i >= len(track_matches.values):
                # 防止索引越界
                track['lifetime'] -= 1
                if track['lifetime'] > 0:
                    new_tracks.append(track)
                continue
                
            best_iou = track_matches.values[i]
            best_det_idx = track_matches.indices[i]
            
            # 检查IoU阈值和类别匹配
            if (best_iou > self.iou_threshold and 
                int(detections[best_det_idx, 4]) == track['cls'] and
                not matched_detections[best_det_idx]):
                
                # 更新轨迹
                track['boxes'].append(detections[best_det_idx, 0:4])
                track['confs'].append(detections[best_det_idx, 5].item())
                track['hit_cnt'] += 1
                track['lifetime'] = self.max_age  # 重置生命周期
                new_tracks.append(track)
                matched_detections[best_det_idx] = True
            else:
                # 未匹配，减少生命周期
                track['lifetime'] -= 1
                if track['lifetime'] > 0:
                    new_tracks.append(track)

        # 为未匹配的检测创建新轨迹
        for i, det in enumerate(detections):
            if not matched_detections[i]:
                new_track = {
                    'id': self.track_id,
                    'boxes': [det[0:4]],
                    'cls': int(det[4]),
                    'confs': [det[5].item()],
                    'hit_cnt': 1,
                    'lifetime': self.max_age
                }
                new_tracks.append(new_track)
                self.track_id += 1

        self.active_tracks = new_tracks

    def finalize(self):
        """
        在视频结束时调用，把剩下的轨迹整理进 finished_tracks
        """
        for track in self.active_tracks:
            if track['hit_cnt'] >= self.min_hits:
                pass  # 可以添加轨迹完成处理
        self.active_tracks = []

    def reset(self):
        self.active_tracks = []
        self.track_id = 0


def calculate_iou_matrix(bbox1, bbox2, device='cpu'):
    """
    :param bbox1: (n1, 4)
    :param bbox2: (n2, 4)
    :param device: device to perform calculations on
    return: IoU (n1, n2)
    """
    # 只有在需要时才进行设备迁移
    if bbox1.device != torch.device(device):
        bbox1 = bbox1.to(device)
    if bbox2.device != torch.device(device):
        bbox2 = bbox2.to(device)
    
    # (n1,1,4) 和 (1,n2,4)
    b1 = bbox1[:, None, :]
    b2 = bbox2[None, :, :]

    # split
    x1_1, y1_1, x2_1, y2_1 = b1[..., 0], b1[..., 1], b1[..., 2], b1[..., 3]
    x1_2, y1_2, x2_2, y2_2 = b2[..., 0], b2[..., 1], b2[..., 2], b2[..., 3]

    # inter
    inter_w = (torch.min(x2_1, x2_2) - torch.max(x1_1, x1_2)).clamp(min=0)
    inter_h = (torch.min(y2_1, y2_2) - torch.max(y1_1, y1_2)).clamp(min=0)
    inter = inter_w * inter_h

    # union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)   # (n1,1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)   # (1,n2)
    union = area1 + area2 - inter + 1e-8

    return inter / union   # (n1,n2)
if __name__ == '__main__':
    tracker = IoUTracker(min_hits=2, iou_threshold=0.3, max_age=2)

    # simulate YOLO bboxes
    class BBoxes:
        def __init__(self, boxes, cls, conf):
            self.xyxy = torch.tensor(boxes, dtype=torch.float32)  # (N,4)
            self.cls = torch.tensor(cls, dtype=torch.int64)  # (N,)
            self.conf = torch.tensor(conf, dtype=torch.float32)  # (N,)

    # simulate 5 frames
    frames = [
        BBoxes(boxes=[(10, 10, 50, 50)], cls=[0], conf=[0.9]),
        BBoxes(boxes=[(12, 12, 52, 52)], cls=[0], conf=[0.85]),
        BBoxes(boxes=[(14, 14, 54, 54)], cls=[0], conf=[0.8]),
        BBoxes(boxes=[(200, 200, 240, 240)], cls=[1], conf=[0.95]),
        BBoxes(boxes=[], cls=[], conf=[])
    ]

    for i, bboxes in enumerate(frames, 1):
        print(f"\nFrame {i}, 输入: {bboxes.xyxy.tolist()}")
        tracker.update(bboxes)

    # finalize
    tracker.finalize()

    print("\n=== 最终输出的轨迹 ===")
    for t in tracker.finished_tracks:
        print(f"TrackID {t['id']} 命中 {t['hit_cnt']} 次, 轨迹: {t['boxes']}")
    # from ultralytics import YOLO
    # model = YOLO('../models/yolov8n.pt')
    # results = model.predict(source=r'../MOT17/MOT17/test/MOT17-01-DPM/img1/000001.jpg')
    # bboxes = results[0].boxes
    # tracker = IoUTracker()
    # tracker.update(bboxes)

    # # test_iou_matrix
    # def test_calculate_iou_matrix():
    #     """测试calculate_iou_matrix函数并打印详细输出"""
    #     print("===== 测试1: 形状相同的边界框 =====")
    #     bbox1 = torch.tensor([
    #         [10, 10, 50, 50],  # 框1
    #         [20, 20, 60, 60]  # 框2
    #     ], dtype=torch.float32)
    #
    #     bbox2 = torch.tensor([
    #         [15, 15, 35, 35],  # 框A
    #         [25, 25, 55, 55]  # 框B
    #     ], dtype=torch.float32)
    #
    #     iou_matrix = calculate_iou_matrix(bbox1, bbox2)
    #     print("输入bbox1形状:", bbox1.shape)
    #     print("输入bbox1:\n", bbox1)
    #     print("\n输入bbox2形状:", bbox2.shape)
    #     print("输入bbox2:\n", bbox2)
    #     print("\n生成的IoU矩阵形状:", iou_matrix.shape)
    #     print("IoU矩阵:\n", iou_matrix)
    #     print("\n" + "=" * 50 + "\n")
    #
    #     print("===== 测试2: 形状不同的边界框 =====")
    #     bbox3 = torch.tensor([
    #         [0, 0, 10, 10]  # 单个框
    #     ], dtype=torch.float32)
    #
    #     bbox4 = torch.tensor([
    #         [2, 2, 8, 8],  # 框C
    #         [5, 5, 15, 15],  # 框D
    #         [20, 20, 30, 30]  # 框E（无交集）
    #     ], dtype=torch.float32)
    #
    #     iou_matrix2 = calculate_iou_matrix(bbox3, bbox4)
    #     print("输入bbox3形状:", bbox3.shape)
    #     print("输入bbox3:\n", bbox3)
    #     print("\n输入bbox4形状:", bbox4.shape)
    #     print("输入bbox4:\n", bbox4)
    #     print("\n生成的IoU矩阵形状:", iou_matrix2.shape)
    #     print("IoU矩阵:\n", iou_matrix2)
    #     print("\n" + "=" * 50 + "\n")
    #
    #     print("===== 测试3: 无交集的边界框 =====")
    #     bbox5 = torch.tensor([
    #         [0, 0, 10, 10],
    #         [20, 20, 30, 30]
    #     ], dtype=torch.float32)
    #
    #     bbox6 = torch.tensor([
    #         [11, 11, 21, 21],
    #         [31, 31, 41, 41]
    #     ], dtype=torch.float32)
    #
    #     iou_matrix3 = calculate_iou_matrix(bbox5, bbox6)
    #     print("输入bbox5:\n", bbox5)
    #     print("输入bbox6:\n", bbox6)
    #     print("IoU矩阵:\n", iou_matrix3)
    #
    #
    # if __name__ == "__main__":
    #     test_calculate_iou_matrix()

