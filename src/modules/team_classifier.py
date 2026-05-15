import numpy as np
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
from sports.common.team import TeamClassifier

# Usando nossa Camada de Configuração em Memória
from modules.common import load_config

class TeamClassifierWrapper:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        
        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            self.device = 'cpu'
            print("MPS not available, falling back to CPU")
        
        self.classifier = None
        
        # Centralizando o uso de paths a partir do config.yaml
        from modules.common import get_project_root
        self.model_path = get_project_root() / "outputs" / "team_classifier.joblib"

        
        self._load_if_exists()
        print(f"TeamClassifier initialized on device: {self.device}")

    def _load_if_exists(self):
        """Load pre-trained classifier if it exists."""
        if self.model_path.exists():
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
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
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
        Atribui goleiros aos times baseado na proximidade da Mediana Espacial (Centro do bloco defensivo).
        Operação 100% vetorizada para máxima performance.
        """
        if len(goalkeepers_xy) == 0:
            return np.array([], dtype=int)
        
        team_0_mask = players_team_ids == 0
        team_1_mask = players_team_ids == 1
        
        # Uso de Mediana (Estado da arte para blocos defensivos) em vez de Média
        if np.sum(team_0_mask) == 0:
            median_0 = np.median(players_xy[team_1_mask], axis=0) if np.sum(team_1_mask) > 0 else np.array([0.0, 0.0])
        else:
            median_0 = np.median(players_xy[team_0_mask], axis=0)
            
        if np.sum(team_1_mask) == 0:
            median_1 = np.median(players_xy[team_0_mask], axis=0) if np.sum(team_0_mask) > 0 else np.array([0.0, 0.0])
        else:
            median_1 = np.median(players_xy[team_1_mask], axis=0)
            
        # Vetorização (Broadcasting) do cálculo de distância
        # Forma: (M, 2) - (2,) -> (M, 2) -> norm -> (M,)
        dist_0 = np.linalg.norm(goalkeepers_xy - median_0, axis=1)
        dist_1 = np.linalg.norm(goalkeepers_xy - median_1, axis=1)
        
        # Retorna 0 se dist_0 < dist_1, senão 1 (Operação binária direta na memória em C)
        gk_team_ids = np.where(dist_0 < dist_1, 0, 1)
        
        return gk_team_ids


if __name__ == "__main__":
    classifier = TeamClassifierWrapper()
    print("Module test passed. TeamClassifier ready.")