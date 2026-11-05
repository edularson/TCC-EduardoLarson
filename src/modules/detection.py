import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Importando a nossa Camada de Configuração Otimizada (Memória Cache)
from modules.common import load_config

class FootballDetector:
    def __init__(self, config_path=None):
        # I/O zero-latency: Carrega direto da memória
        self.config = load_config(config_path)

        # Set device (MPS for M4 Pro, fallback to CPU)
        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = 'cpu'

        # Load detection model (local YOLOv8 weights)
        model_path = self.config['models']['detection']['path']
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Modelo especializado em bola — treinado com SoccerNet Tracking
        ball_model_path = self.config['models'].get('ball_detection', {}).get('path')
        if ball_model_path and Path(ball_model_path).exists():
            self.ball_model = YOLO(ball_model_path)
            self.ball_model.to(self.device)
            print(f"Ball model carregado: {ball_model_path}")
        else:
            self.ball_model = None
            print("Ball model nao encontrado — usando modelo principal para bola.")

        # Class mapping (confirmed: ball:0, goalkeeper:1, player:2, referee:3)
        self.class_ids = self.config['models']['detection']['class_ids']
        self.conf_threshold = self.config['models']['detection']['confidence_threshold']
        self.nms_threshold = self.config['models']['detection']['nms_threshold']
        self.imgsz = self.config['models']['detection']['imgsz']

        print(f"FootballDetector initialized on device: {self.device}")
        print(f"Model: {model_path}, Classes: {self.config['models']['detection']['class_names']}")

    def infer(self, frame: np.ndarray) -> any:
        """
        Run YOLOv8 inference on a single frame.
        Returns Ultralytics Results object.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        return results[0]  # Single frame, take first result

    def get_detections(self, frame: np.ndarray) -> tuple:
        result = self.infer(frame)
        if result.boxes is None or len(result.boxes.cls) == 0:
            balls_yolo = np.empty((0, 5))
        else:
            boxes_data = result.boxes.data.cpu().numpy()
            classes    = boxes_data[:, 5].astype(int)
            mask_ball  = (classes == self.class_ids['ball'])
            mask_player = (classes == self.class_ids['player'])
            mask_gk    = (classes == self.class_ids['goalkeeper'])
            mask_ref   = (classes == self.class_ids['referee'])

            players     = boxes_data[mask_player]
            goalkeepers = boxes_data[mask_gk]
            referees    = boxes_data[mask_ref]
            balls_yolo  = boxes_data[mask_ball][:, :5]

        # ── Detector especializado de bola (SAHI) ────────────────────────────
        if self.ball_model is not None:
            h_img, w_img = frame.shape[:2]
            overlap  = 100
            mid_w    = w_img // 2
            conf_th  = self.config['models']['ball_detection'].get('confidence_threshold', 0.25)

            res_left  = self.ball_model.predict(frame[:, :mid_w + overlap],
                            conf=conf_th, imgsz=self.imgsz,
                            device=self.device, verbose=False)[0]
            res_right = self.ball_model.predict(frame[:, mid_w - overlap:],
                            conf=conf_th, imgsz=self.imgsz,
                            device=self.device, verbose=False)[0]

            ball_list = []
            if res_left.boxes is not None and len(res_left.boxes) > 0:
                ball_list.append(res_left.boxes.data.cpu().numpy()[:, :5])
            if res_right.boxes is not None and len(res_right.boxes) > 0:
                rb = res_right.boxes.data.cpu().numpy()[:, :5]
                rb[:, 0] += (mid_w - overlap)
                rb[:, 2] += (mid_w - overlap)
                ball_list.append(rb)

            if ball_list:
                balls = np.vstack(ball_list)
                if len(balls) > 1:
                    boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in balls]
                    keep = cv2.dnn.NMSBoxes(boxes_xywh, balls[:, 4].tolist(),
                                            score_threshold=0.0, nms_threshold=0.3)
                    balls = balls[keep.flatten()] if len(keep) > 0 else np.empty((0, 5))
            else:
                balls = np.empty((0, 5))
        else:
            balls = balls_yolo

        # ── Filtro geométrico ─────────────────────────────────────────────────
        if len(balls) > 0:
            h_img, w_img = frame.shape[:2]
            areas        = (balls[:, 2] - balls[:, 0]) * (balls[:, 3] - balls[:, 1])
            widths       = balls[:, 2] - balls[:, 0]
            heights      = balls[:, 3] - balls[:, 1]
            aspect_ratio = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-6)
            cx           = (balls[:, 0] + balls[:, 2]) / 2
            cy           = (balls[:, 1] + balls[:, 3]) / 2
            margin_top   = int(h_img * 0.20)
            margin_side  = int(w_img * 0.02)
            valid_mask   = (
                (areas <= 350) &
                (aspect_ratio <= 2.0) &
                (cy > margin_top) &
                (cx > margin_side) &
                (cx < w_img - margin_side)
            )
            balls = balls[valid_mask]

        # Garante que players/goalkeepers/referees existem mesmo sem boxes
        if result.boxes is None or len(result.boxes.cls) == 0:
            players     = np.empty((0, 6))
            goalkeepers = np.empty((0, 6))
            referees    = np.empty((0, 6))

        return balls, players, goalkeepers, referees


    def get_ball_center(self, ball_detection: tuple) -> tuple:
        """Get center (x,y) of ball bbox."""
        x1, y1, x2, y2, _ = ball_detection
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_player_foot_position(self, player_detection: tuple) -> tuple:
        """Get bottom-center (foot position) of player bbox."""
        x1, y1, x2, y2, *_ = player_detection
        return ((x1 + x2) // 2, y2)


if __name__ == "__main__":
    # Test detection module
    detector = FootballDetector()
    print("Module test passed. Detector ready.")
