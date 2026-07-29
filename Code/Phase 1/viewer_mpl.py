"""
Single-window live viewer (OpenCV-only).

Shows one 'S-MSCKF Viewer' window combining:
  * left panel  — top-down XY trajectory (GT red, estimate black, latest blue)
  * right panel — live left-camera frame
  * bottom bar  — pose count + sequence time

GT is expected already in the filter frame (vio.py does the Umeyama alignment).
"""
import time
import threading
from queue import Queue

import numpy as np
import cv2


WIN_W = 1280
WIN_H = 560
LEFT_W = 560
PAD = 24
BAR_H = 36
PLOT_H = WIN_H - BAR_H

RED = (0, 0, 255)         # BGR
BLACK = (20, 20, 20)
BLUE = (220, 80, 40)
GRID = (225, 225, 225)
AXIS = (170, 170, 170)
BG = (250, 250, 250)
TEXT = (55, 55, 55)


def _build_to_px(gt):
    pts = gt if gt is not None else np.zeros((1, 3))
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    span = max(x_max - x_min, y_max - y_min, 1e-3)
    pad = 0.15 * span
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad
    scale = min((LEFT_W - 2 * PAD) / (x_max - x_min),
                (PLOT_H - 2 * PAD) / (y_max - y_min))

    def to_px(x, y):
        u = int(PAD + (x - x_min) * scale)
        v = int(PLOT_H - PAD - (y - y_min) * scale)
        return u, v

    return to_px, (x_min, x_max, y_min, y_max), scale


def _draw_plot_chrome(canvas, to_px, bounds, scale):
    x_min, x_max, y_min, y_max = bounds
    # faint grid
    step = 1.0  # 1 m
    x0 = np.floor(x_min / step) * step
    y0 = np.floor(y_min / step) * step
    x = x0
    while x <= x_max:
        u, _ = to_px(x, 0)
        cv2.line(canvas, (u, PAD), (u, PLOT_H - PAD), GRID, 1, cv2.LINE_AA)
        x += step
    y = y0
    while y <= y_max:
        _, v = to_px(0, y)
        cv2.line(canvas, (PAD, v), (LEFT_W - PAD, v), GRID, 1, cv2.LINE_AA)
        y += step
    # border
    cv2.rectangle(canvas, (PAD, PAD), (LEFT_W - PAD, PLOT_H - PAD), AXIS, 1)
    # 1m scale bar
    p1 = to_px(x_min + 0.06 * (x_max - x_min), y_min + 0.06 * (y_max - y_min))
    p2 = to_px(x_min + 0.06 * (x_max - x_min) + 1.0, y_min + 0.06 * (y_max - y_min))
    cv2.line(canvas, p1, p2, TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, '1 m', (p1[0], p1[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    # axis labels
    cv2.putText(canvas, 'X [m]', (LEFT_W - 60, PLOT_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Y [m]', (6, PAD + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    # title
    cv2.putText(canvas, 'Trajectory (top-down)', (PAD, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1, cv2.LINE_AA)
    # legend
    lx = LEFT_W - 210
    cv2.line(canvas, (lx, 14), (lx + 28, 14), RED, 3, cv2.LINE_AA)
    cv2.putText(canvas, 'Ground truth', (lx + 36, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    cv2.line(canvas, (lx, 30), (lx + 28, 30), BLACK, 3, cv2.LINE_AA)
    cv2.putText(canvas, 'MSCKF estimate', (lx + 36, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)


def _compose(traj_canvas, img_panel, bar_text):
    frame = np.full((WIN_H, WIN_W, 3), BG, dtype=np.uint8)
    frame[:PLOT_H, :LEFT_W] = traj_canvas
    # right panel: camera area
    rx0 = LEFT_W
    right_w = WIN_W - LEFT_W
    cv2.putText(frame, 'Live cam0', (rx0 + PAD, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1, cv2.LINE_AA)
    if img_panel is not None:
        h, w = img_panel.shape[:2]
        avail_h = PLOT_H - 2 * PAD
        avail_w = right_w - 2 * PAD
        s = min(avail_w / w, avail_h / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(img_panel, (nw, nh))
        ox = rx0 + PAD + (avail_w - nw) // 2
        oy = PAD + (avail_h - nh) // 2
        frame[oy:oy + nh, ox:ox + nw] = resized
    # separator
    cv2.line(frame, (LEFT_W, PAD // 2), (LEFT_W, PLOT_H - PAD // 2),
             (210, 210, 210), 1)
    # status bar
    cv2.rectangle(frame, (0, PLOT_H), (WIN_W, WIN_H), (240, 240, 240), -1)
    cv2.line(frame, (0, PLOT_H), (WIN_W, PLOT_H), (200, 200, 200), 1)
    cv2.putText(frame, bar_text, (PAD, PLOT_H + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, 'press q to quit',
                (WIN_W - 160, PLOT_H + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv2.LINE_AA)
    return frame


class Viewer(object):
    def __init__(self, gt_trajectory=None):
        # Pose is tiny (4x4 float64) — small unbounded queue is fine.
        self.pose_queue = Queue()
        # Image is large; keep only the latest frame in a lock-guarded slot
        # so a slow consumer can't build backlog and freeze the panel.
        self._img_lock = threading.Lock()
        self._latest_img = None
        self.gt_trajectory = (
            np.asarray(gt_trajectory, dtype=np.float64)
            if gt_trajectory is not None else None)

        # Thread (not Process) so numpy arrays pass by reference — no pickling
        # across a pipe, no backpressure-induced stalls.
        self.view_thread = threading.Thread(target=self._run)
        self.view_thread.daemon = True
        self.view_thread.start()

    def update_pose(self, pose):
        if pose is None:
            return
        self.pose_queue.put(np.asarray(pose.matrix(), dtype=np.float64))

    def update_image(self, image):
        if image is None:
            return
        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        with self._img_lock:
            self._latest_img = image

    def _run(self):
        pose_queue = self.pose_queue
        gt = self.gt_trajectory
        to_px, bounds, scale = _build_to_px(gt)

        base = np.full((PLOT_H, LEFT_W, 3), BG, dtype=np.uint8)
        _draw_plot_chrome(base, to_px, bounds, scale)
        if gt is not None and len(gt) >= 2:
            gt_px = np.array([to_px(p[0], p[1]) for p in gt])
            cv2.polylines(base, [gt_px], isClosed=False,
                          color=RED, thickness=2, lineType=cv2.LINE_AA)

        cv2.namedWindow('S-MSCKF Viewer', cv2.WINDOW_AUTOSIZE)

        traj_px = []
        t_start = time.time()

        while True:
            while not pose_queue.empty():
                try:
                    item = pose_queue.get_nowait()
                except Exception:
                    break
                pose = item
                x, y, _z = pose[:3, 3]
                traj_px.append(to_px(x, y))

            with self._img_lock:
                latest_img = self._latest_img

            canvas = base.copy()
            if len(traj_px) >= 2:
                arr = np.array(traj_px, dtype=np.int32)
                cv2.polylines(canvas, [arr], isClosed=False,
                              color=BLACK, thickness=2, lineType=cv2.LINE_AA)
            if traj_px:
                u, v = traj_px[-1]
                cv2.rectangle(canvas, (u - 5, v - 5), (u + 5, v + 5),
                              BLUE, -1)

            img_panel = None
            if latest_img is not None:
                img_panel = (cv2.cvtColor(latest_img, cv2.COLOR_RGB2BGR)
                             if latest_img.ndim == 3 else latest_img)

            elapsed = time.time() - t_start
            bar = f'poses: {len(traj_px):4d}    elapsed: {elapsed:6.1f} s'

            frame = _compose(canvas, img_panel, bar)
            cv2.imshow('S-MSCKF Viewer', frame)

            key = cv2.waitKey(15) & 0xFF
            if key in (ord('q'), 27):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    import time
    t = np.linspace(0, 4 * np.pi, 300)
    gt = np.stack([np.cos(t) * 3, np.sin(t) * 3, np.zeros_like(t)], axis=1)
    v = Viewer(gt_trajectory=gt)
    time.sleep(3)
