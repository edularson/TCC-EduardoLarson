import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib
from pathlib import Path
import ast

class XGModel:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_type = self.config['xg']['model_type']
        self.model = None

        # Coordenadas StatsBomb: campo 105x68m, gol direito em x=120
        self.goal_center = np.array([120.0, 40.0])
        self.goal_top_post = np.array([120.0, 44.0])
        self.goal_bottom_post = np.array([120.0, 36.0])

        print(f"XGModel initialized. Model type: {self.model_type}")

    def _pitch_120_to_statsbomb(self, shot_pos: np.ndarray) -> np.ndarray:
        """
        Converte coordenadas do SoccerPitchConfiguration (120x70m)
        para coordenadas StatsBomb (120x80m -> normalizado para 105x68m).
        Na prática: escala x de [0,120] para [0,120] e y de [0,70] para [0,80].
        """
        # SoccerPitchConfig: 120x70m | StatsBomb: 120x80m
        x = shot_pos[0]
        y = shot_pos[1] * (80.0 / 70.0)
        return np.array([x, y])

    def extract_features(self, shot_pos: np.ndarray, body_part: int = 1) -> np.ndarray:
        """
        Extrai features de xG para um chute detectado no vídeo.
        shot_pos: coordenadas no espaço SoccerPitchConfiguration (120x70m)
        body_part: 0=left_foot, 1=right_foot, 2=head
        """
        # Converte para espaço StatsBomb antes de calcular geometria
        pos = self._pitch_120_to_statsbomb(shot_pos)

        # Distância ao gol
        shot_distance = np.linalg.norm(pos - self.goal_center)

        # Ângulo entre os postes
        v1 = self.goal_top_post - pos
        v2 = self.goal_bottom_post - pos
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        shot_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

        # Body part one-hot
        body_onehot = np.zeros(3)
        body_onehot[body_part] = 1

        return np.array([shot_distance, shot_angle] + list(body_onehot))

    def load_statsbomb_data(self, data_path: str) -> tuple:
        """Carrega dados do StatsBomb e extrai apenas features disponíveis no vídeo."""
        df = pd.read_csv(data_path)
        shots_df = df[df['type'] == 'Shot'].copy()

        shots_df['label'] = shots_df['shot_outcome'].apply(
            lambda x: 1 if isinstance(x, str) and 'Goal' in x else 0
        )

        def parse_location(loc):
            try:
                coords = ast.literal_eval(loc)
                return float(coords[0]), float(coords[1])
            except:
                return None, None

        shots_df[['loc_x', 'loc_y']] = shots_df['location'].apply(
            lambda l: pd.Series(parse_location(l))
        )
        shots_df = shots_df.dropna(subset=['loc_x', 'loc_y'])

        def encode_body_part(bp):
            if isinstance(bp, str):
                if 'Head' in bp: return 2
                elif 'Left' in bp: return 0
            return 1

        features = []
        for _, row in shots_df.iterrows():
            # StatsBomb já usa 120x80m, não precisa converter
            shot_pos_sb = np.array([row['loc_x'], row['loc_y']])
            body_part = encode_body_part(row.get('shot_body_part'))

            # Calcula distância e ângulo diretamente no espaço StatsBomb
            shot_distance = np.linalg.norm(shot_pos_sb - self.goal_center)
            v1 = self.goal_top_post - shot_pos_sb
            v2 = self.goal_bottom_post - shot_pos_sb
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            shot_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

            body_onehot = np.zeros(3)
            body_onehot[body_part] = 1

            feat = np.array([shot_distance, shot_angle] + list(body_onehot))
            features.append(feat)

        X = np.array(features)
        y = shots_df['label'].values

        print(f"Loaded {len(X)} shots | Goals: {y.sum()} ({y.mean()*100:.1f}%) | Features: {X.shape[1]}")
        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        self.model.fit(X_train, y_train)
        print(f"XGModel trained on {len(X_train)} samples. Model: {self.model_type}")

    def predict_xg(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        y_pred_proba = self.predict_xg(X_test)
        return {
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }

    def save_model(self, path=None):
        if path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            path = project_root / "outputs" / "xg_model.joblib"
        if self.model is None:
            raise ValueError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model}, path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        if path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            path = project_root / "outputs" / "xg_model.joblib"
        checkpoint = joblib.load(path)
        self.model = checkpoint['model']
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Test xG module
    xg = XGModel()
    
    # Test feature extraction
    shot_pos = np.array([100.0, 34.0])  # 5m from goal
    features = xg.extract_features(shot_pos, body_part=1, def_pressure=2)
    print(f"Test features: {features}")
    
    # Test with dummy data
    X_dummy = np.array([features, [90.0, 34.0, 0, 1, 0, 1, 3]])  # Two shots
    y_dummy = np.array([1, 0])  # One goal, one no goal
    xg.train(X_dummy, y_dummy)
    xg_pred = xg.predict_xg(X_dummy)
    print(f"Test xG predictions: {xg_pred}")
    
    metrics = xg.evaluate(X_dummy, y_dummy)
    print(f"Test metrics: {metrics}")
    print("Module test passed. XGModel ready.")
