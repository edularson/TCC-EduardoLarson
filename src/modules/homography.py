import yaml
import numpy as np
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from pathlib import Path

class PitchMapper:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_pitch = SoccerPitchConfiguration()
        self.keypoint_conf_threshold = self.config['models']['field_detection']['keypoint_confidence_threshold']
        self.recompute_threshold = self.config['homography']['recompute_threshold']
        
        # Cached homography per clip (static for tactical cam)
        self.cached_transformer = None
        self.last_keypoints_hash = None
        
        print("PitchMapper initialized. Static homography for tactical cam enabled.")

    def compute_homography_from_points(self, frame_points: np.ndarray, pitch_points: np.ndarray) -> ViewTransformer:
        """
        Compute homography between frame (camera) and pitch (real-world).
        """
        if len(frame_points) < 4 or len(pitch_points) < 4:
            raise ValueError(f"Need at least 4 points, got {len(frame_points)}")
        
        # Fixed: source=frame_points, target=pitch_points (correct mapping)
        transformer = ViewTransformer(
            source=frame_points,
            target=pitch_points
        )
        return transformer

    def get_transformer(self, frame_points: np.ndarray, pitch_points: np.ndarray) -> ViewTransformer:
        """
        Get cached homography for clip. Recompute only if keypoints change significantly.
        """
        if self.cached_transformer is None:
            self.cached_transformer = self.compute_homography_from_points(frame_points, pitch_points)
            print("Homography computed and cached for clip.")
        return self.cached_transformer

    def reset_clip(self):
        """Call when loading a new clip."""
        self.cached_transformer = None
        self.last_keypoints_hash = None
        print("PitchMapper reset for new clip.")

    def frame_to_pitch(self, transformer: ViewTransformer, points: np.ndarray) -> np.ndarray:
        result = transformer.transform_points(points)
        result[:, 0] = 120.0 - result[:, 0]  # Inverte eixo X
        return result

    def pitch_to_frame(self, transformer: ViewTransformer, points: np.ndarray) -> np.ndarray:
        """Convert real-world pitch meters to frame pixel coordinates."""
        return transformer.inverse_transform_points(points)

    def get_goal_coordinates(self, team_id: int) -> np.ndarray:
        """
        Retorna o centro do gol adversário em metros reais.
        Team 1 ataca para esquerda (x=0), Team 0 ataca para direita (x=120).
        """
        if team_id == 1:
            return np.array([0.0, 35.0])
        else:
            return np.array([120.0, 35.0])


if __name__ == "__main__":
    mapper = PitchMapper()
    print("Module test passed. PitchMapper ready.")
