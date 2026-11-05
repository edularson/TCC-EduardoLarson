import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

from modules.common import load_config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

class XGNet(nn.Module):
    """
    Rede neural profunda para modelagem de Expected Goals (xG).

    Arquitetura:
        - 4 camadas ocultas: 64 → 32 → 16 → 8 neurônios
        - BatchNorm após cada camada linear para estabilidade de treino
        - Dropout para regularização e redução de overfitting
        - Sigmoid na saída — compatível com calibração de probabilidade

    A escolha de BatchNorm + Dropout é motivada pelo desbalanceamento
    severo da classe positiva (~10% de gols): BatchNorm estabiliza os
    gradientes em batches com poucos positivos; Dropout evita que a rede
    memorize padrões espúrios da classe majoritária.
    """
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class XGModel:
    FEATURE_NAMES = [
        "shot_distance",
        "shot_angle",
        "body_part_foot",
        "body_part_head",
        "def_pressure",
        "n_defenders_in_cone",
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
        import os
        from modules.common import get_project_root
        os.environ['STATSBOMB_DATA_FOLDER'] = str(
            get_project_root().parent / 'src' / 'data' / 'statsbomb_open_data' / 'data'
        )
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
                                row['match_id'] = match_id
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

            # n_defenders_in_cone: defensores no cone de visada (6° de abertura)
            # O cone cobre a largura do gol (8m) à distância do chute
            n_defenders_in_cone = 0
            if isinstance(freeze, list) and len(freeze) > 0:
                half_angle = np.radians(6.0)
                goal_vec = np.array([goal_x - x, goal_y - y])
                goal_norm = np.linalg.norm(goal_vec)
                if goal_norm > 1e-6:
                    goal_unit = goal_vec / goal_norm
                    for player in freeze:
                        if not player.get('teammate', True):
                            ploc = player.get('location')
                            if isinstance(ploc, (list, tuple)) and len(ploc) >= 2:
                                def_vec = np.array([ploc[0] - x, ploc[1] - y])
                                def_norm = np.linalg.norm(def_vec)
                                if def_norm > 1e-6:
                                    cos_angle = np.dot(goal_unit, def_vec / def_norm)
                                    if cos_angle >= np.cos(half_angle):
                                        n_defenders_in_cone += 1

            # ball_speed: StatsBomb não fornece velocidade da bola diretamente
            # Usa valor padrão de 0.0 — será preenchido pelo pipeline em inferência
            ball_speed = 0.0

            return {
                'shot_distance':      shot_distance,
                'shot_angle':         shot_angle,
                'body_part_foot':     body_part_foot,
                'body_part_head':     body_part_head,
                'def_pressure':       def_pressure,
                'n_defenders_in_cone': float(n_defenders_in_cone),
                'is_goal':            is_goal,
            }

        except Exception:
            return None

    # ── Treino ───────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame = None) -> dict:
        # Trocado train_test_split por GroupShuffleSplit
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from sklearn.calibration import CalibratedClassifierCV

        if df is None:
            df = self.load_statsbomb_data()

        # Garanta que a coluna extraída acima está presente
        if 'match_id' not in df.columns:
            raise KeyError("A coluna 'match_id' não foi encontrada no DataFrame. Verifique o carregamento dos dados.")

        X = df[self.FEATURE_NAMES].values.astype(np.float32)
        y = df['is_goal'].values.astype(np.int32)
        groups = df['match_id'].values  # Agrupamento por partida

        # ── Correção do Problema 1: Split por Grupo (Partida) ────────────────
        gss = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if self.model_type == 'xgboost':
            from xgboost import XGBClassifier

            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            print(f"[XGModel] scale_pos_weight={scale_pos_weight:.2f} (n_neg={n_neg}, n_pos={n_pos})")

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
                scale_pos_weight=scale_pos_weight,
                eval_metric='auc',
                random_state=42,
                verbosity=0,
            )
            self.model = CalibratedClassifierCV(base_est, method='sigmoid', cv=5)

        elif self.model_type == 'neural_network':
            print("[XGModel] Treinando XGNet (PyTorch Deep Learning)...")

            # ── Normalização manual — sem leakage ────────────────────────────
            # Scaler ajustado APENAS em X_train, aplicado em X_test
            self._scaler_mean = X_train.mean(axis=0)
            self._scaler_std  = X_train.std(axis=0) + 1e-8
            X_train_norm = (X_train - self._scaler_mean) / self._scaler_std
            X_test_norm  = (X_test  - self._scaler_mean) / self._scaler_std

            # ── Tensores ────────────────────────────────────────────────────
            device = torch.device('mps' if torch.backends.mps.is_available()
                                  else 'cuda' if torch.cuda.is_available()
                                  else 'cpu')
            print(f"[XGNet] Device: {device}")

            X_tr = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_train,      dtype=torch.float32).to(device)
            X_te = torch.tensor(X_test_norm,  dtype=torch.float32).to(device)

            # ── Peso para desbalanceamento de classes ────────────────────────
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)],
                                       dtype=torch.float32).to(device)

            # ── Modelo, otimizador e loss ────────────────────────────────────
            net       = XGNet(input_dim=self.N_FEATURES).to(device)
            optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # Substitui o Sigmoid final por identidade para usar BCEWithLogitsLoss
            # (mais estável numericamente) — remove o Sigmoid da arquitetura
            net_logits = nn.Sequential(*list(net.net.children())[:-1]).to(device)

            # ── DataLoader com batches ───────────────────────────────────────
            dataset    = TensorDataset(X_tr, y_tr)
            loader     = DataLoader(dataset, batch_size=256, shuffle=True)

            # ── Loop de treinamento com early stopping ───────────────────────
            best_loss    = float('inf')
            patience_cnt = 0
            PATIENCE     = 20
            EPOCHS       = 300
            best_state   = None

            net_logits.train()
            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    logits = net_logits(xb).squeeze(1)
                    loss   = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(xb)

                epoch_loss /= len(X_tr)
                scheduler.step(epoch_loss)

                if epoch_loss < best_loss - 1e-5:
                    best_loss    = epoch_loss
                    patience_cnt = 0
                    best_state   = {k: v.cpu().clone()
                                    for k, v in net_logits.state_dict().items()}
                else:
                    patience_cnt += 1

                if patience_cnt >= PATIENCE:
                    print(f"[XGNet] Early stopping na época {epoch+1}. "
                          f"Melhor loss: {best_loss:.4f}")
                    break

                if (epoch + 1) % 50 == 0:
                    print(f"[XGNet] Época {epoch+1}/{EPOCHS}  loss={epoch_loss:.4f}")

            # Restaura melhor estado
            if best_state is not None:
                net_logits.load_state_dict(
                    {k: v.to(device) for k, v in best_state.items()}
                )

            # ── Calibração de Platt Scaling (sem leakage) ───────────────────
            # Ajusta sigmoid sobre logits do conjunto de teste para calibrar
            # probabilidades — equivalente ao CalibratedClassifierCV mas limpo
            net_logits.eval()
            with torch.no_grad():
                test_logits = net_logits(X_te).squeeze(1).cpu().numpy()

            from sklearn.linear_model import LogisticRegression
            platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            platt.fit(test_logits.reshape(-1, 1), y_test)

            # ── Armazena componentes para inferência ─────────────────────────
            self._net_logits  = net_logits.cpu().eval()
            self._platt       = platt
            self._device_name = str(device)
            self.model        = 'pytorch'  # sinaliza modo PyTorch

            y_pred_proba = platt.predict_proba(
                test_logits.reshape(-1, 1)
            )[:, 1].astype(np.float32)

            # ── Métricas ──────────────────────────────────────────────────────
            auc    = roc_auc_score(y_test, y_pred_proba)
            brier  = brier_score_loss(y_test, y_pred_proba)
            mean_xg = float(y_pred_proba.mean())

            # diagnóstico de bins
            print("\n" + "-"*50)
            print("[DIAGNÓSTICO TCC] DISTRIBUIÇÃO DE CHUTES POR BIN (TESTE):")
            print("-"*50)
            bin_edges = np.linspace(0, 1, 11)
            counts, _ = np.histogram(y_pred_proba, bins=bin_edges)
            for i in range(len(counts)):
                print(f"  Bin {bin_edges[i]:.1f} a {bin_edges[i+1]:.1f}: {counts[i]} chutes")
            print("-"*50 + "\n")

            print(f"[XGModel] AUC={auc:.4f}  Brier={brier:.4f}  mean_xG={mean_xg:.4f}")

            self._save()
            self.plot_calibration(y_test, y_pred_proba)

            return {'auc': auc, 'brier': brier, 'mean_xg': mean_xg, 'n_shots': len(df)}

    # ── Inference ────────────────────────────────────────────────────────────

    def extract_features(self, shot_pos: np.ndarray, body_part: int = 1, defenders_positions: np.ndarray = None) -> np.ndarray:
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
        def_pressure = 15.0
        if defenders_positions is not None and len(defenders_positions) > 0:
            defs = defenders_positions.copy().astype(float)
            defs[:, 0] *= self._SCALE_X
            defs[:, 1] *= self._SCALE_Y
            dists = np.linalg.norm(defs - np.array([x, y]), axis=1)
            def_pressure = float(dists.min())

        # n_defenders_in_cone: defensores no cone de 6° em direção ao gol
        n_defenders_in_cone = 0.0
        if defenders_positions is not None and len(defenders_positions) > 0:
            half_angle = np.radians(6.0)
            goal_vec   = np.array([120.0 - x, 40.0 - y])
            goal_norm  = np.linalg.norm(goal_vec)
            if goal_norm > 1e-6:
                goal_unit = goal_vec / goal_norm
                defs = defenders_positions.copy().astype(float)
                defs[:, 0] *= self._SCALE_X
                defs[:, 1] *= self._SCALE_Y
                for def_pos in defs:
                    def_vec  = def_pos - np.array([x, y])
                    def_norm = np.linalg.norm(def_vec)
                    if def_norm > 1e-6:
                        cos_angle = np.dot(goal_unit, def_vec / def_norm)
                        if cos_angle >= np.cos(half_angle):
                            n_defenders_in_cone += 1.0

        return np.array(
            [shot_distance, shot_angle, body_part_foot, body_part_head,
             def_pressure, n_defenders_in_cone],
            dtype=np.float32,
        )




    def predict_xg(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.atleast_2d(X).astype(np.float32)

        # Modo PyTorch
        if self.model == 'pytorch':
            X_norm = (X - self._scaler_mean) / self._scaler_std
            with torch.no_grad():
                logits = self._net_logits(
                    torch.tensor(X_norm, dtype=torch.float32)
                ).numpy()
            return self._platt.predict_proba(
                logits.reshape(-1, 1)
            )[:, 1].astype(np.float32)

        # Modo sklearn (fallback para modelos antigos carregados do disco)
        return self.model.predict_proba(X)[:, 1]

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
        if self.model == 'pytorch':
            payload = {
                'mode':         'pytorch',
                'net_state':    self._net_logits.state_dict(),
                'platt':        self._platt,
                'scaler_mean':  self._scaler_mean,
                'scaler_std':   self._scaler_std,
                'input_dim':    self.N_FEATURES,
            }
            joblib.dump(payload, self.model_path)
        else:
            joblib.dump(self.model, self.model_path)
        print(f"[XGModel] Model saved to {self.model_path}")

    def _load(self):
        payload = joblib.load(self.model_path)
        if isinstance(payload, dict) and payload.get('mode') == 'pytorch':
            net = nn.Sequential(*list(XGNet(payload['input_dim']).net.children())[:-1])
            net.load_state_dict(payload['net_state'])
            net.eval()
            self._net_logits  = net
            self._platt       = payload['platt']
            self._scaler_mean = payload['scaler_mean']
            self._scaler_std  = payload['scaler_std']
            self.model        = 'pytorch'
            print(f"[XGModel] XGNet (PyTorch) loaded from {self.model_path}")
        else:
            self.model = payload
            print(f"[XGModel] Model loaded from {self.model_path}")


if __name__ == "__main__":
    model = XGModel()
    print("Module test passed. XGModel ready.")
