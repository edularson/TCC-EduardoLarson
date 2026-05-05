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
        """
        Convert frame pixel coordinates to real-world pitch meters.
        Args:
            transformer: ViewTransformer instance
            points: (N, 2) array of (x,y) pixel coordinates
        Returns:
            (N, 2) array of (x,y) real-world meters
        """
        return transformer.transform_points(points)

    def pitch_to_frame(self, transformer: ViewTransformer, points: np.ndarray) -> np.ndarray:
        """Convert real-world pitch meters to frame pixel coordinates."""
        return transformer.inverse_transform_points(points)

    def get_goal_coordinates(self, team_id: int) -> np.ndarray:
        """
        Get real-world coordinates of both goals.
        Returns:
            team_0_goal: (2,) center of goal for team 0
            team_1_goal: (2,) center of goal for team 1
        """
        # SoccerPitchConfiguration: goal vertices
        left_goal_vertices = np.array([
            self.config_pitch.vertices[24],  # Goal left top
            self.config_pitch.vertices[25]   # Goal left bottom
        ])
        right_goal_vertices = np.array([
            self.config_pitch.vertices[26],  # Goal right top
            self.config_pitch.vertices[27]   # Goal right bottom
        ])
        
        left_goal_center = np.mean(left_goal_vertices, axis=0)
        right_goal_center = np.mean(right_goal_vertices, axis=0)
        
        # Team 0 defends left goal, Team 1 defends right goal
        if team_id == 0:
            return left_goal_center
        else:
            return right_goal_center


if __name__ == "__main__":
    mapper = PitchMapper()
    print("Module test passed. PitchMapper ready.")
