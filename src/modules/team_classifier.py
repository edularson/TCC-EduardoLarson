import yaml
import numpy as np
from sports.common.team import TeamClassifier
from tqdm import tqdm
from pathlib import Path
import joblib

class TeamClassifierWrapper:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['device']
        if self.device == 'mps' and not __import__('torch').backends.mps.is_available():
            self.device = 'cpu'
        
        self.classifier = None
        self.model_path = str(Path(__file__).parent.parent / "outputs" / "team_classifier.joblib")
        self._load_if_exists()
        print(f"TeamClassifier initialized on device: {self.device}")

    def _load_if_exists(self):
        """Load pre-trained classifier if it exists."""
        if Path(self.model_path).exists():
            try:
                checkpoint = joblib.load(self.model_path)
                self.classifier = checkpoint['classifier']
                print(f"Loaded pre-trained classifier from {self.model_path}")
            except Exception as e:
                print(f"Error loading classifier: {e}")

    def train(self, crops: list, force_retrain: bool = False):
        """
        Train team classifier from player crops.
        Saves to disk to avoid re-training.
        """
        if self.classifier is not None and not force_retrain:
            print("Classifier already trained. Use force_retrain=True to retrain.")
            return
        
        if len(crops) < 10:
            raise ValueError(f"Need at least 10 crops to train, got {len(crops)}")
        
        print(f"Training TeamClassifier on {len(crops)} crops...")
        self.classifier = TeamClassifier(device=self.device)
        self.classifier.fit(crops)
        
        # Save to disk
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'classifier': self.classifier}, self.model_path)
        print(f"Classifier saved to {self.model_path}")

    def predict(self, crops: list) -> np.ndarray:
        """Predict team IDs for player crops."""
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        return self.classifier.predict(crops)

    def resolve_goalkeepers(self, 
                            players_xy: np.ndarray,
                            players_team_ids: np.ndarray,
                            goalkeepers_xy: np.ndarray) -> np.ndarray:
        """
        Assign goalkeepers to teams based on proximity to team centroids.
        Fixed: handle empty teams to avoid NaN.
        """
        if len(goalkeepers_xy) == 0:
            return np.array([])
        
        # Calculate team centroids (handle empty teams)
        team_0_mask = players_team_ids == 0
        team_1_mask = players_team_ids == 1
        
        if np.sum(team_0_mask) == 0:
            # No team 0 players, use team 1 centroid
            team_0_centroid = np.mean(players_xy[team_1_mask], axis=0) if np.sum(team_1_mask) > 0 else np.array([0.0, 0.0])
        else:
            team_0_centroid = np.mean(players_xy[team_0_mask], axis=0)
        
        if np.sum(team_1_mask) == 0:
            # No team 1 players, use team 0 centroid
            team_1_centroid = np.mean(players_xy[team_0_mask], axis=0) if np.sum(team_0_mask) > 0 else np.array([0.0, 0.0])
        else:
            team_1_centroid = np.mean(players_xy[team_1_mask], axis=0)
        
        gk_team_ids = []
        for gk_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
            dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
            gk_team_ids.append(0 if dist_0 < dist_1 else 1)
        
        return np.array(gk_team_ids)


if __name__ == "__main__":
    # Test module
    classifier = TeamClassifierWrapper()
    print("Module test passed. TeamClassifier ready.")
