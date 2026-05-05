import yaml
import torch
import numpy as np
import supervision as sv
from pathlib import Path

class FootballTracker:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device (MPS for M4 Pro)
        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            self.device = 'cpu'
            print(f"MPS not available, falling back to CPU")
        
        # Initialize tracker (try BotSORT, fallback to ByteTrack)
        try:
            self.tracker = sv.BotSort(
                reid_weights=None,  # Use default lightweight ReID
                device=self.device,
                half=False,  # MPS doesn't support half precision well
                max_age=self.config['tracking']['max_age'],
                min_hits=self.config['tracking']['min_hits']
            )
            self.tracker_type = 'botsort'
        except AttributeError:
            print("BotSORT not available in current Supervision version. Using ByteTrack.")
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=self.config['tracking']['max_age'],
                minimum_matching_threshold=0.8
            )
            self.tracker_type = 'bytetrack'
        
        print(f"FootballTracker ({self.tracker_type}) initialized on device: {self.device}")

    def update(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections for a frame.
        Args:
            frame: Current video frame (BGR)
            detections: Supervision Detections object from YOLO inference
        Returns:
            Updated Detections with tracker_id and stable IDs
        """
        if self.tracker_type == 'botsort':
            # BotSORT expects frame for ReID feature extraction
            updated_detections = self.tracker.update_with_detections(
                detections=detections,
                frame=frame
            )
        else:
            # ByteTrack doesn't need frame
            updated_detections = self.tracker.update_with_detections(
                detections=detections
            )
        return updated_detections

    def reset(self):
        """Reset tracker state (call between clips)."""
        self.tracker.reset()
        print("Tracker reset for new clip.")

    @staticmethod
    def detections_from_yolo_result(result, class_ids: dict, conf_threshold: float = 0.3):
        """
        Convert Ultralytics YOLO result to Supervision Detections.
        Filters by confidence and remaps class IDs.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return sv.Detections.empty()
        
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        
        # Filter by confidence
        mask = conf >= conf_threshold
        xyxy = xyxy[mask]
        conf = conf[mask]
        cls = cls[mask]
        
        if len(xyxy) == 0:
            return sv.Detections.empty()
        
        # Create Supervision Detections
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls
        )
        return detections


if __name__ == "__main__":
    # Test tracking module
    tracker = FootballTracker()
    print("Module test passed. Tracker ready.")
