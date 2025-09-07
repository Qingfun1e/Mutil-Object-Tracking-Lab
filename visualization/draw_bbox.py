import cv2
import numpy
import colorsys

def get_color(track_id, total_tracks=20):
    hue = (track_id % total_tracks) / total_tracks
    sat, val = 0.9, 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b*255), int(g*255), int(r*255))  # BGR

class TrajectoryDrawer:
    def __init__(self, frame_shape, alpha=0.6, decay=0.95):
        # 轨迹图层（全黑）
        self.traj_layer = numpy.zeros(frame_shape, dtype=numpy.uint8)
        self.alpha = alpha   # 叠加透明度
        self.decay = decay   # 衰减系数（0.95 表示旧轨迹逐渐淡化）

    def draw(self, tracker, frame_to_draw, min_hits, cls_list, fps=None, is_paused=False, play_speed=1.0):
        # 轨迹图层做轻微衰减（让旧线条逐渐消失）
        self.traj_layer = cv2.addWeighted(self.traj_layer, self.decay, numpy.zeros_like(self.traj_layer), 0, 0)

        for track in tracker.active_tracks:
            if track['hit_cnt'] > min_hits:
                # 当前框
                x1, y1, x2, y2 = map(int, track['boxes'][-1])
                cls_id = cls_list[track['cls']]
                conf = track['confs'][-1]
                track_id = track['id']
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                color = get_color(track_id)

                # 绘制目标框和信息
                cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_to_draw, f'ID:{track_id} Cls:{cls_id} Conf:{conf:.2f}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(frame_to_draw, (cx, cy), 3, color, -1)

                # 在轨迹层上追加一条线
                if len(track['boxes']) > 1:
                    prev_box = track['boxes'][-2]
                    px, py = int((prev_box[0] + prev_box[2]) / 2), int((prev_box[1] + prev_box[3]) / 2)
                    cv2.line(self.traj_layer, (px, py), (cx, cy), color, 2)

        # 合并轨迹层
        frame_to_draw = cv2.addWeighted(frame_to_draw, 1.0, self.traj_layer, self.alpha, 0)

        # 左上角显示 FPS
        if fps is not None:
            cv2.putText(frame_to_draw, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 左上角显示播放状态和速度（在FPS下方）
        status_text = f"Speed: {play_speed:.2f}x"
        if is_paused:
            status_text = "PAUSED | " + status_text
        cv2.putText(frame_to_draw, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 左上角显示操作提示（在状态信息下方）
        help_text = "Keys: SPACE=Pause/Resume  ->=Step  +/-=Speed  0=ResetSpeed  R=ResetTracker  Q/ESC=Quit  C=Capture"
        cv2.putText(frame_to_draw, help_text, (10, frame_to_draw.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame_to_draw