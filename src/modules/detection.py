import yaml
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class FootballDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            # Get absolute path to config (module location -> project root -> configs/config.yaml)
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device (MPS for M4 Pro, fallback to CPU)
        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            print(f"MPS not available, falling back to CPU")
            self.device = 'cpu'
        
        # Load detection model (local YOLOv8 weights)
        model_path = self.config['models']['detection']['path']
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
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
        """
        Returns:
          - balls: list of (x1,y1,x2,y2,conf)
          - players: list of (x1,y1,x2,y2,conf,class_id)
          - goalkeepers: list of (x1,y1,x2,y2,conf,class_id)
          - referees: list of (x1,y1,x2,y2,conf,class_id)
        """
        result = self.infer(frame)
        if result.boxes is None or len(result.boxes) == 0:
            return [], [], [], []
        
        boxes = result.boxes
        balls, players, goalkeepers, referees = [], [], [], []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_id == self.class_ids['ball']:
                balls.append((x1, y1, x2, y2, conf))
            elif cls_id == self.class_ids['player']:
                players.append((x1, y1, x2, y2, conf, cls_id))
            elif cls_id == self.class_ids['goalkeeper']:
                goalkeepers.append((x1, y1, x2, y2, conf, cls_id))
            elif cls_id == self.class_ids['referee']:
                referees.append((x1, y1, x2, y2, conf, cls_id))
        
        return balls, players, goalkeepers, referees

    def filter_low_confidence(self, detections: list, conf_threshold: float = None) -> list:
        """Filter detections by confidence threshold."""
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        return [d for d in detections if d[4] >= conf_threshold]

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
