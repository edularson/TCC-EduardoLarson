import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

class XGModel:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_type = self.config['xg']['model_type']
        self.feature_names = self.config['xg']['features']
        self.statsbomb_comp = self.config['xg']['statsbomb_competition']
        self.statsbomb_seasons = self.config['xg']['statsbomb_season']
        
        self.model = None
        self.goal_center = np.array([105.0, 34.0])  # FIFA pitch: 105x68m, right goal center
        self.goal_left_post = np.array([105.0, 34.0 + 7.32/2])  # Right goal top (7.32m width)
        self.goal_right_post = np.array([105.0, 34.0 - 7.32/2])  # Right goal bottom
        
        print(f"XGModel initialized. Model type: {self.model_type}")

    def extract_features(self, shot_pos: np.ndarray, body_part: int, def_pressure: int) -> np.ndarray:
        """
        Extract xG features for a single detected shot.
        Args:
            shot_pos: (2,) real-world (x,y) meters of shot position
            body_part: 0=left_foot, 1=right_foot, 2=head
            def_pressure: number of opponents within 5m
        Returns:
            (4,) array: [shot_distance, shot_angle, body_part_onehot, def_pressure]
        """
        # 1. Shot distance to goal
        shot_distance = np.linalg.norm(shot_pos - self.goal_center)
        
        # 2. Shot angle (degrees between shot and goalposts)
        v1 = self.goal_left_post - shot_pos
        v2 = self.goal_right_post - shot_pos
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        shot_angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        
        # 3. Body part one-hot
        body_onehot = np.zeros(3)
        body_onehot[body_part] = 1
        
        # 4. Defensive pressure
        return np.array([shot_distance, shot_angle] + list(body_onehot) + [def_pressure])

    def load_statsbomb_data(self, data_path: str) -> tuple:
        import ast

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
            return 1  # Right Foot default

        def encode_technique(t):
            mapping = {'Normal Shot': 0, 'Volley': 1, 'Half Volley': 2,
                    'Overhead Kick': 3, 'Diving Header': 4, 'Lob': 5}
            return mapping.get(t, 0) if isinstance(t, str) else 0

        def encode_shot_type(t):
            mapping = {'Open Play': 0, 'Free Kick': 1, 'Corner': 2, 'Penalty': 3}
            return mapping.get(t, 0) if isinstance(t, str) else 0

        def bool_col(val):
            if val is True or val == 'True': return 1
            return 0

        features = []
        for _, row in shots_df.iterrows():
            shot_pos = np.array([row['loc_x'], row['loc_y']])
            body_part = encode_body_part(row.get('shot_body_part'))

            # Base geometric features
            base = self.extract_features(shot_pos, body_part, def_pressure=0)

            # Extra features
            technique  = encode_technique(row.get('shot_technique'))
            shot_type  = encode_shot_type(row.get('shot_type'))
            one_on_one = bool_col(row.get('shot_one_on_one'))
            open_goal  = bool_col(row.get('shot_open_goal'))
            first_time = bool_col(row.get('shot_first_time'))
            pressure   = bool_col(row.get('under_pressure'))

            feat = np.append(base, [technique, shot_type, one_on_one, open_goal, first_time, pressure])
            features.append(feat)

        X = np.array(features)
        y = shots_df['label'].values

        print(f"Loaded {len(X)} shots | Goals: {y.sum()} ({y.mean()*100:.1f}%) | Features: {X.shape[1]}")
        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost binary classifier for xG."""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        elif self.model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.model.fit(X_train, y_train)
        print(f"XGModel trained on {len(X_train)} samples. Model: {self.model_type}")

    def predict_xg(self, X: np.ndarray) -> np.ndarray:
        """Predict xG (goal probability) for input features."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]  # Probability of class 1 (goal)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate xG model with TCC-required metrics."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred_proba = self.predict_xg(X_test)
        metrics = {
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        return metrics

    def save_model(self, path=None):
        """Save trained model to disk."""
        if path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            path = project_root / "outputs" / "xg_model.joblib"
        
        if self.model is None:
            raise ValueError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, path)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load trained model from disk."""
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
