import yaml
import torch
import numpy as np
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
from pathlib import Path
import cv2

class FieldDetector:
    def __init__(self, config_path=None):
        module_dir = Path(__file__).parent
        project_root = module_dir.parent

        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device (MPS for M4 Pro)
        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            self.device = 'cpu'
            print(f"MPS not available, falling back to CPU")

        # Load your trained keypoint model
        self.model_path = str(project_root / "TCC_VarzeaVision_Keypoint_Backup/datasets/runs/pose/train/weights/best.pt")
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Loaded trained keypoint model: {self.model_path}")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            print("Falling back to yolov8x-pose-p6.pt...")
            self.model = YOLO("yolov8x-pose-p6.pt")
            self.model.to(self.device)
        
        self.conf_threshold = self.config['models']['field_detection']['confidence_threshold']
        self.keypoint_conf_threshold = self.config['models']['field_detection']['keypoint_confidence_threshold']
        self.config_pitch = SoccerPitchConfiguration()
        
        print(f"FieldDetector initialized on device: {self.device}")
        print(f"Keypoint conf threshold: {self.keypoint_conf_threshold}")

    def infer(self, frame: np.ndarray) -> any:
        """
        Run YOLOv8-pose inference for pitch keypoints.
        Returns Ultralytics Results object with keypoints.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=1280,
            device=self.device,
            verbose=False
        )
        return results[0]


    def get_keypoints(self, frame: np.ndarray) -> tuple:
        """
        Extract valid pitch keypoints from frame.
        Returns:
            - frame_points: (N, 2) array of valid keypoints in frame pixels
            - pitch_points: (N, 2) array of corresponding real-world pitch points
            - valid_mask: boolean mask for valid keypoints
        """
        result = self.infer(frame)
        
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return np.array([]), np.array([]), np.array([])
        
        if result.keypoints.conf is None or len(result.keypoints.conf) == 0:
            return np.array([]), np.array([]), np.array([])

        # Pega a detecção com maior confiança média
        mean_confs = result.keypoints.conf.mean(dim=1)
        best_idx = mean_confs.argmax().item()

        kpts = result.keypoints.xy[best_idx].cpu().numpy()       # (32, 2)
        kpts_conf = result.keypoints.conf[best_idx].cpu().numpy() # (32,)

        valid_mask = kpts_conf >= self.keypoint_conf_threshold

        if np.sum(valid_mask) < 4:
            return np.array([]), np.array([]), valid_mask

        vertices = np.array(self.config_pitch.vertices) / 100.0
        valid_indices = np.where(valid_mask)[0]
        valid_indices = valid_indices[valid_indices < len(vertices)]

        if len(valid_indices) < 4:
            return np.array([]), np.array([]), valid_mask

        frame_points = kpts[valid_indices]
        pitch_points = vertices[valid_indices]

        return frame_points, pitch_points, valid_mask

    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Draw detected keypoints on frame for debugging."""
        annotated = frame.copy()
        if len(keypoints) == 0:
            return annotated
        
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 165, 255), -1)  # Orange circles
            if mask is not None and i < len(mask):
                label = f"K{i}"
                cv2.putText(annotated, label, (int(x)+10, int(y)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated


if __name__ == "__main__":
    detector = FieldDetector()
    print("Module test passed. FieldDetector ready.")
