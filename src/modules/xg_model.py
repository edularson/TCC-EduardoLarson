import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

from modules.common import load_config

warnings.filterwarnings('ignore')


class XGModel:
    FEATURE_NAMES = [
        "shot_distance",
        "shot_angle",
        "body_part_foot",
        "body_part_head",
        "def_pressure",
    ]
    N_FEATURES = len(FEATURE_NAMES)

    # ── C8: fatores de escala campo real (UEFA) → StatsBomb ──────────────────
    # UEFA:      105m × 68m
    # StatsBomb: 120m × 80m
    # O modelo foi treinado com coordenadas StatsBomb; o inference recebe metros
    # reais. Ambos os eixos precisam ser normalizados antes de calcular distância
    # e ângulo — anteriormente só Y era normalizado (e com valor errado: 70m).
    _SCALE_X = 120.0 / 105.0   # ≈ 1.1429
    _SCALE_Y =  80.0 /  68.0   # ≈ 1.1765

    def __init__(self, config_path=None):
        self.config = load_config(config_path)

        xg_cfg = self.config.get('xg', {})
        self.model_type = xg_cfg.get('model_type', 'xgboost')
        self.competitions = xg_cfg.get('statsbomb_competitions', [])

        from modules.common import get_project_root

        runs_dir = get_project_root() / "runs"
        self.model_path = runs_dir / "xg_model.joblib"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = None

        print(f"[XGModel] Looking for model at: {self.model_path}")
        if self.model_path.exists():
            self._load()
        else:
            print(f"[XGModel] No saved model at {self.model_path}. Call train() first.")

    # ── Carregamento de dados ────────────────────────────────────────────────

    def load_statsbomb_data(self) -> pd.DataFrame:
        try:
            from statsbombpy import sb
        except ImportError:
            raise ImportError("Install statsbombpy: pip install statsbombpy")

        all_shots = []

        for comp in self.competitions:
            cid = comp['competition_id']
            sid = comp['season_id']

            try:
                matches = sb.matches(competition_id=cid, season_id=sid)
                if matches is None or len(matches) == 0:
                    continue

                print(f"  [xG] Loading comp={cid} season={sid} ({len(matches)} matches)...")

                for match_id in matches['match_id'].tolist():
                    try:
                        events = sb.events(match_id=match_id)
                        shots = events[events['type'] == 'Shot']
                        if len(shots) == 0:
                            continue

                        for shot in shots.to_dict('records'):
                            row = self._parse_statsbomb_shot(shot)
                            if row is not None:
                                all_shots.append(row)

                    except Exception as e:
                        print(f"    [xG] Skipping match {match_id}: {e}")
                        continue

            except Exception as e:
                print(f"  [xG] Failed comp={cid} season={sid}: {e}")
                continue

        if len(all_shots) == 0:
            raise ValueError("No shot data loaded.")

        df = pd.DataFrame(all_shots)
        print(f"\n[XGModel] Total shots loaded: {len(df)}")
        print(f"[XGModel] Goal rate: {df['is_goal'].mean():.1%}  "
              f"(goals={df['is_goal'].sum()}, non-goals={len(df)-df['is_goal'].sum()})")
        return df

    def _parse_statsbomb_shot(self, shot: dict) -> dict | None:
        try:
            loc = shot.get('location')
            if not loc or len(loc) < 2:
                return None

            x, y = float(loc[0]), float(loc[1])
            # Coordenadas já estão em sistema StatsBomb (120×80) — sem escala aqui
            goal_x, goal_y = 120.0, 40.0

            shot_distance = float(np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2))

            goal_left_y, goal_right_y = 36.0, 44.0
            angle_left  = np.arctan2(goal_left_y  - y, goal_x - x)
            angle_right = np.arctan2(goal_right_y - y, goal_x - x)
            shot_angle  = float(abs(np.degrees(angle_right - angle_left)))

            bp_raw = shot.get('shot_body_part', '')
            if isinstance(bp_raw, dict):
                bp_raw = bp_raw.get('name', '')
            bp_raw = str(bp_raw).lower()
            body_part_foot = 1.0 if ('foot' in bp_raw or 'left' in bp_raw or 'right' in bp_raw) else 0.0
            body_part_head = 1.0 if 'head' in bp_raw else 0.0

            def_pressure = 0.0
            freeze = shot.get('shot_freeze_frame')
            if isinstance(freeze, list) and len(freeze) > 0:
                min_def_dist = float('inf')
                for player in freeze:
                    if not player.get('teammate', True):
                        ploc = player.get('location')
                        if isinstance(ploc, (list, tuple)) and len(ploc) >= 2:
                            dist = float(np.sqrt((ploc[0] - x) ** 2 + (ploc[1] - y) ** 2))
                            min_def_dist = min(min_def_dist, dist)
                if min_def_dist < float('inf'):
                    def_pressure = min_def_dist

            outcome = shot.get('shot_outcome', '')
            if isinstance(outcome, dict):
                outcome = outcome.get('name', '')
            is_goal = 1 if str(outcome).lower() == 'goal' else 0

            return {
                'shot_distance':  shot_distance,
                'shot_angle':     shot_angle,
                'body_part_foot': body_part_foot,
                'body_part_head': body_part_head,
                'def_pressure':   def_pressure,
                'is_goal':        is_goal,
            }

        except Exception:
            return None

    # ── Treino ───────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame = None) -> dict:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from sklearn.calibration import CalibratedClassifierCV

        if df is None:
            df = self.load_statsbomb_data()

        X = df[self.FEATURE_NAMES].values.astype(np.float32)
        y = df['is_goal'].values.astype(np.int32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if self.model_type == 'xgboost':
            from xgboost import XGBClassifier

            # ── C9: scale_pos_weight calculado a partir da distribuição real ──
            # Proporção correta: n_negativos / n_positivos.
            # Hardcodar 2.0 subestimava a desproporção real (~10-15% de gols),
            # fazendo o modelo subestimar xG sistematicamente.
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            print(f"[XGModel] scale_pos_weight={scale_pos_weight:.2f} "
                  f"(n_neg={n_neg}, n_pos={n_pos})")

            base_est = XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.6,
                min_child_weight=10,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,   # ← dinâmico
                eval_metric='auc',
                random_state=42,
                verbosity=0,
            )
            self.model = CalibratedClassifierCV(base_est, method='sigmoid', cv=5)

        self.model.fit(X_train, y_train)

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc      = roc_auc_score(y_test, y_pred_proba)
        brier    = brier_score_loss(y_test, y_pred_proba)
        mean_xg  = float(y_pred_proba.mean())

        print(f"[XGModel] AUC={auc:.4f}  Brier={brier:.4f}  mean_xG={mean_xg:.4f}")

        self._save()
        self.plot_calibration(y_test, y_pred_proba)

        return {'auc': auc, 'brier': brier, 'mean_xg': mean_xg, 'n_shots': len(df)}

    # ── Inference ────────────────────────────────────────────────────────────

    def extract_features(self,
                         shot_pos: np.ndarray,
                         body_part: int = 1,
                         defenders_positions: np.ndarray = None) -> np.ndarray:
        """
        Extrai features a partir de coordenadas do campo real (metros, UEFA 105×68).
        Converte internamente para o sistema StatsBomb (120×80) antes de calcular
        distância e ângulo, garantindo compatibilidade com o espaço de treino.

        Args:
            shot_pos: posição do chute em metros, sistema pipeline (origem canto esq).
                      X ∈ [0, 105], Y ∈ [0, 68].
            body_part: 1 = pé (default), 0 = cabeça.
            defenders_positions: array (N, 2) posições dos defensores em metros reais.

        Returns:
            Array float32 com as 5 features no espaço de treino.
        """
        # ── C8: normalização de ambos os eixos ───────────────────────────────
        x = float(shot_pos[0]) * self._SCALE_X
        y = float(shot_pos[1]) * self._SCALE_Y

        goal_x, goal_y = 120.0, 40.0
        shot_distance = float(np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2))

        goal_left_y, goal_right_y = 36.0, 44.0
        angle_left  = np.arctan2(goal_left_y  - y, goal_x - x)
        angle_right = np.arctan2(goal_right_y - y, goal_x - x)
        shot_angle  = float(abs(np.degrees(angle_right - angle_left)))

        body_part_foot = 1.0 if body_part != 0 else 0.0
        body_part_head = 1.0 if body_part == 0 else 0.0

        # def_pressure: mesma escala StatsBomb para compatibilidade com treino
        def_pressure = 0.0
        if defenders_positions is not None and len(defenders_positions) > 0:
            defs = defenders_positions.copy().astype(float)
            defs[:, 0] *= self._SCALE_X
            defs[:, 1] *= self._SCALE_Y
            dists = np.linalg.norm(defs - np.array([x, y]), axis=1)
            def_pressure = float(dists.min())

        return np.array(
            [shot_distance, shot_angle, body_part_foot, body_part_head, def_pressure],
            dtype=np.float32,
        )

    def predict_xg(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(np.atleast_2d(X).astype(np.float32))[:, 1]

    # ── Diagnóstico ──────────────────────────────────────────────────────────

    def plot_calibration(self, y_test: np.ndarray, y_probs: np.ndarray,
                         save_path: str = "outputs/CurvaCalibracao_xG.png"):
        """
        Salva a curva de calibração em disco.
        Não usa plt.show() para não bloquear execuções batch/headless.
        """
        import matplotlib
        matplotlib.use('Agg')   # backend sem display
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2,
                color='#e74c3c', label='VarzeaVision xG')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray',
                label='Calibração Perfeita (y=x)')
        ax.set_xlabel('Probabilidade Média Prevista de xG')
        ax.set_ylabel('Frequência Real de Gols')
        ax.set_title('Curva de Calibração')
        ax.legend()
        ax.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[XGModel] Calibration curve saved to {save_path}")

    # ── Persistência ─────────────────────────────────────────────────────────

    def _save(self):
        joblib.dump(self.model, self.model_path)
        print(f"[XGModel] Model saved to {self.model_path}")

    def _load(self):
        self.model = joblib.load(self.model_path)
        print(f"[XGModel] Model loaded from {self.model_path}")


if __name__ == "__main__":
    model = XGModel()
    print("Module test passed. XGModel ready.")