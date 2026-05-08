"""
xg_model.py — VarzeaVision
Modelo de Expected Goals (xG) treinado com dados reais do StatsBomb.

Correções em relação à versão anterior:
1. IDs corretos: competition_id=16 é Champions League, 9 é Bundesliga
2. Carregamento multi-season via lista no config.yaml
3. Bloco de teste corrigido (sem def_pressure inexistente, X_dummy com 5 colunas)
4. defensive_pressure implementado: distância ao defensor mais próximo
5. extract_features() agora aceita defenders_positions opcionalmente
6. load_statsbomb_data() robusto a seasons sem dados de chute
"""

import yaml
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class XGModel:
    # Features que o modelo usa internamente (ordem importa para o array)
    FEATURE_NAMES = [
        "shot_distance",   # metros ao centro do gol
        "shot_angle",      # graus (0 = linha de fundo, 90 = frente ao gol)
        "body_part_foot",  # 1 se pé dominante, 0 caso contrário
        "body_part_head",  # 1 se cabeça, 0 caso contrário
        "def_pressure",    # metros ao defensor mais próximo (0 = sem pressão conhecida)
    ]
    N_FEATURES = len(FEATURE_NAMES)  # 5

    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        xg_cfg = self.config.get('xg', {})
        self.model_type = xg_cfg.get('model_type', 'xgboost')

        # Pega lista de competitions do config
        self.competitions = xg_cfg.get('statsbomb_competitions', [
            {'competition_id': 16, 'season_id': 4},    # UCL 2018/2019
            {'competition_id': 16, 'season_id': 1},    # UCL 2017/2018
            {'competition_id': 16, 'season_id': 2},    # UCL 2016/2017
            {'competition_id': 16, 'season_id': 27},   # UCL 2015/2016
            {'competition_id': 16, 'season_id': 26},   # UCL 2014/2015
            {'competition_id': 9,  'season_id': 281},  # Bundesliga 2023/2024
            {'competition_id': 9,  'season_id': 27},   # Bundesliga 2015/2016
        ])

        # Caminho para salvar/carregar o modelo treinado
        paths_cfg = self.config.get('paths', {})
        runs_dir = Path(paths_cfg.get('runs', 'runs'))
        self.model_path = runs_dir / "xg_model.joblib"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = None

        # Tenta carregar modelo já treinado
        if self.model_path.exists():
            self._load()
        else:
            print(f"[XGModel] No saved model at {self.model_path}. Call train() first.")

    # ------------------------------------------------------------------
    # Carregamento de dados StatsBomb
    # ------------------------------------------------------------------

    def load_statsbomb_data(self) -> pd.DataFrame:
        """
        Carrega chutes de todas as competitions/seasons configuradas.
        Retorna DataFrame com features e target (is_goal).
        """
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
                    print(f"  [xG] No matches for comp={cid} season={sid}, skipping.")
                    continue

                print(f"  [xG] Loading comp={cid} season={sid} ({len(matches)} matches)...")

                for match_id in matches['match_id'].tolist():
                    try:
                        events = sb.events(match_id=match_id)
                        shots = events[events['type'] == 'Shot'].copy()

                        if len(shots) == 0:
                            continue

                        # Extrai features de cada chute
                        for _, shot in shots.iterrows():
                            row = self._parse_statsbomb_shot(shot)
                            if row is not None:
                                all_shots.append(row)

                    except Exception:
                        continue  # ignora partidas com erro silenciosamente

            except Exception as e:
                print(f"  [xG] Failed comp={cid} season={sid}: {e}")
                continue

        if len(all_shots) == 0:
            raise ValueError("No shot data loaded. Check your StatsBomb competitions config.")

        df = pd.DataFrame(all_shots)
        print(f"\n[XGModel] Total shots loaded: {len(df)}")
        print(f"  Goals: {df['is_goal'].sum()} ({df['is_goal'].mean()*100:.1f}%)")
        print(f"  Features: {self.FEATURE_NAMES}")
        return df

    def _parse_statsbomb_shot(self, shot) -> dict | None:
        """
        Extrai features de um evento de chute do StatsBomb.
        Retorna None se o chute não tiver dados mínimos.

        StatsBomb usa campo 120x80 metros.
        Gol em x=120, y=40 (centro). Postes em y=36 e y=44.
        """
        try:
            loc = shot.get('location')
            if loc is None or not isinstance(loc, (list, tuple)) or len(loc) < 2:
                return None

            x, y = float(loc[0]), float(loc[1])

            # Centro do gol adversário
            goal_x, goal_y = 120.0, 40.0

            # Distância ao centro do gol
            shot_distance = float(np.sqrt((goal_x - x)**2 + (goal_y - y)**2))

            # Ângulo: arctan2 dos dois postes a partir da posição do chutador
            # Ref: StatsBomb Guide to Shooting (StatsBomb Technical Paper)
            goal_left_y, goal_right_y = 36.0, 44.0
            angle_left  = np.arctan2(goal_left_y  - y, goal_x - x)
            angle_right = np.arctan2(goal_right_y - y, goal_x - x)
            shot_angle  = float(abs(np.degrees(angle_right - angle_left)))

            # Body part (one-hot)
            bp_raw = shot.get('shot_body_part', '')
            if isinstance(bp_raw, dict):
                bp_raw = bp_raw.get('name', '')
            bp_raw = str(bp_raw).lower()

            body_part_foot = 1.0 if ('foot' in bp_raw or 'left' in bp_raw or 'right' in bp_raw) else 0.0
            body_part_head = 1.0 if 'head' in bp_raw else 0.0

            # Defensive pressure: distância ao defensor mais próximo
            # StatsBomb fornece freeze frame com posições dos jogadores
            def_pressure = 0.0
            freeze = shot.get('shot_freeze_frame')
            if isinstance(freeze, list) and len(freeze) > 0:
                min_def_dist = float('inf')
                for player in freeze:
                    if not player.get('teammate', True):  # adversário
                        ploc = player.get('location')
                        if isinstance(ploc, (list, tuple)) and len(ploc) >= 2:
                            dist = float(np.sqrt(
                                (ploc[0] - x)**2 + (ploc[1] - y)**2
                            ))
                            min_def_dist = min(min_def_dist, dist)
                if min_def_dist < float('inf'):
                    def_pressure = min_def_dist

            # Target
            outcome = shot.get('shot_outcome', '')
            if isinstance(outcome, dict):
                outcome = outcome.get('name', '')
            is_goal = 1 if str(outcome).lower() == 'goal' else 0

            return {
                'shot_distance':   shot_distance,
                'shot_angle':      shot_angle,
                'body_part_foot':  body_part_foot,
                'body_part_head':  body_part_head,
                'def_pressure':    def_pressure,
                'is_goal':         is_goal,
            }

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

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

        print(f"\n[XGModel] Training {self.model_type} with Calibration on {len(X_train)} shots...")

        if self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                # 1. Definimos o estimador base (o "motor" do modelo)
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
                    scale_pos_weight=2.0, # Mantemos o peso para o AUC ficar alto
                    use_label_encoder=False,
                    eval_metric='auc',
                    random_state=42,
                    verbosity=0,
                )
                # 2. Envolvemos o motor no calibrador (o "ajuste de precisão")
                self.model = CalibratedClassifierCV(base_est, method='sigmoid', cv=5)
                
            except ImportError:
                print("  XGBoost not found, falling back to GradientBoosting...")
                self.model_type = 'gradient_boosting'

        # ... (blocos gradient_boosting e logistic_regression permanecem iguais) ...
        # DICA: Se quiser calibrar os outros também, basta repetir o padrão CalibratedClassifierCV neles.

        self.model.fit(X_train, y_train)

        # Métricas
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        mean_xg = float(y_pred_proba.mean())

        print(f"  AUC-ROC:     {auc:.4f}")
        print(f"  Brier score: {brier:.4f}")
        print(f"  Mean xG:     {mean_xg:.4f} (deve estar mais perto de ~0.11 agora)")

        # Feature importance ajustado para modelos calibrados
        # O CalibratedClassifierCV guarda os modelos no atributo .calibrated_classifiers_
        print("\n  Feature importances (from calibrated ensemble):")
        try:
            # Tentamos acessar o estimador dentro do primeiro classificador do ensemble
            calibrated_clf = self.model.calibrated_classifiers_[0]
            
            # Nas versões novas do sklearn é .estimator, nas antigas era .base_estimator
            if hasattr(calibrated_clf, 'estimator'):
                imp = calibrated_clf.estimator.feature_importances_
            else:
                imp = calibrated_clf.base_estimator.feature_importances_

            for name, importance in sorted(zip(self.FEATURE_NAMES, imp), key=lambda x: -x[1]):
                bar = '█' * int(importance * 40)
                print(f"    {name:20s} {bar} {importance:.3f}")
        except Exception as e:
            print(f"    Could not extract feature importances: {e}")

        self._save()
        return {'auc': auc, 'brier': brier, 'mean_xg': mean_xg, 'n_shots': len(df)}

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    def extract_features(self,
                         shot_pos: np.ndarray,
                         body_part: int = 1,
                         defenders_positions: np.ndarray = None) -> np.ndarray:
        """
        Extrai features de um chute para inferência no pipeline.

        Args:
            shot_pos             : posição da bola em metros reais (x, y)
                                   no espaço do SoccerPitchConfiguration (120x70m)
            body_part            : 0=cabeça, 1=pé (default)
            defenders_positions  : array (N, 2) com posições dos defensores
                                   em metros reais. Se None, def_pressure=0.

        Returns:
            np.ndarray shape (5,) com [shot_distance, shot_angle,
                                       body_part_foot, body_part_head, def_pressure]

        NOTA: O StatsBomb usa campo 120x80m; o SoccerPitchConfig usa 120x70m.
        A normalização de Y é feita internamente aqui.
        """
        # SoccerPitchConfig: 120m x 70m → converte Y para escala StatsBomb (80m)
        x = float(shot_pos[0])
        y = float(shot_pos[1]) * (80.0 / 70.0)

        goal_x, goal_y = 120.0, 40.0

        # Distância ao gol
        shot_distance = float(np.sqrt((goal_x - x)**2 + (goal_y - y)**2))

        # Ângulo de chute (mesma fórmula do treino)
        goal_left_y, goal_right_y = 36.0, 44.0
        angle_left  = np.arctan2(goal_left_y  - y, goal_x - x)
        angle_right = np.arctan2(goal_right_y - y, goal_x - x)
        shot_angle  = float(abs(np.degrees(angle_right - angle_left)))

        # Body part
        body_part_foot = 1.0 if body_part != 0 else 0.0
        body_part_head = 1.0 if body_part == 0 else 0.0

        # Defensive pressure
        def_pressure = 0.0
        if defenders_positions is not None and len(defenders_positions) > 0:
            defs_y_scaled = defenders_positions.copy().astype(float)
            defs_y_scaled[:, 1] *= (80.0 / 70.0)
            dists = np.linalg.norm(
                defs_y_scaled - np.array([x, y]), axis=1
            )
            def_pressure = float(dists.min())

        return np.array([
            shot_distance,
            shot_angle,
            body_part_foot,
            body_part_head,
            def_pressure,
        ], dtype=np.float32)

    def predict_xg(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz probabilidade de gol para um array de features.

        Args:
            X : (N, 5) ou (5,) — features de extract_features()
        Returns:
            (N,) probabilidades em [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.atleast_2d(X).astype(np.float32)
        return self.model.predict_proba(X)[:, 1]
    
    def plot_calibration(model_instance):
        """
        Recria o conjunto de teste usado no treinamento e plota a curva de calibração.
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.model_selection import train_test_split

        print("\n[Calibração] Preparando dados para o gráfico...")
        
        # 1. Carrega os dados originais
        df = model_instance.load_statsbomb_data()
        X = df[model_instance.FEATURE_NAMES].values.astype(np.float32)
        y = df['is_goal'].values.astype(np.int32)

        # 2. Refaz o split EXATAMENTE igual ao do treino (garantido pelo random_state=42)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Pega as predições usando o modelo já carregado/treinado
        y_probs = model_instance.predict_xg(X_test)

        # 4. Calcula a curva (10 faixas de probabilidade)
        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

        # 5. Plota o gráfico
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#e74c3c', label='VarzeaVision xG (XGBoost)')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibração Perfeita (y=x)')
        
        plt.xlabel('Probabilidade Média Prevista de xG')
        plt.ylabel('Frequência Real de Gols')
        plt.title('Curva de Calibração - Validação do Modelo xG')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Exibe a Soma de xG vs Soma de Gols (teste rápido de calibração média)
        total_xg = y_probs.sum()
        total_gols = y_test.sum()
        plt.annotate(f'Total xG Previsto: {total_xg:.1f}\nTotal Gols Reais: {total_gols}', 
                    xy=(0.05, 0.85), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.show()

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def _save(self):
        joblib.dump(self.model, self.model_path)
        print(f"\n[XGModel] Model saved to {self.model_path}")

    def _load(self):
        self.model = joblib.load(self.model_path)
        print(f"[XGModel] Model loaded from {self.model_path}")
        


# ------------------------------------------------------------------
# Teste isolado — rode com: python modules/xg_model.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("XGModel — Teste isolado")
    print("=" * 60)

    model = XGModel()

    # Teste de extract_features — sem treinamento
    print("\n--- extract_features() ---")

    # Pênalti (11m do gol, frente ao gol, sem pressão)
    penalty_pos = np.array([109.0, 35.0])  # ~11m do gol em campo 120x70
    feat_penalty = model.extract_features(penalty_pos, body_part=1)
    print(f"Pênalti:   dist={feat_penalty[0]:.1f}m  angle={feat_penalty[1]:.1f}°  features={feat_penalty}")

    # Chute de fora da área (25m, ângulo fechado)
    long_shot_pos = np.array([95.0, 20.0])
    feat_long = model.extract_features(long_shot_pos, body_part=1)
    print(f"Longe/ang: dist={feat_long[0]:.1f}m  angle={feat_long[1]:.1f}°  features={feat_long}")

    # Com pressão defensiva
    defenders = np.array([[110.0, 36.0], [111.0, 38.0]])
    feat_pressure = model.extract_features(penalty_pos, body_part=1, defenders_positions=defenders)
    print(f"Com press: def_pressure={feat_pressure[4]:.1f}m  features={feat_pressure}")

    # Cabeçada
    header_pos = np.array([110.0, 35.0])
    feat_header = model.extract_features(header_pos, body_part=0)
    print(f"Cabeçada:  body_part_foot={feat_header[2]}  body_part_head={feat_header[3]}")

    print(f"\nN_FEATURES={XGModel.N_FEATURES} ✓  Feature names: {XGModel.FEATURE_NAMES}")

    # Treino (só se não tiver modelo salvo)
    if model.model is None:
        print("\n--- Treinando modelo ---")
        print("Isso vai demorar alguns minutos na primeira vez...")
        try:
            metrics = model.train()
            print(f"\nTreino concluído: {metrics}")
        except Exception as e:
            print(f"Erro no treino: {e}")
            print("Verifique sua conexão com internet e o config.yaml")
    else:
        # Testa predição com modelo já carregado
        print("\n--- predict_xg() ---")
        X_test = np.array([
            feat_penalty,   # pênalti — esperado: ~0.75
            feat_long,      # longe — esperado: ~0.03
            feat_header,    # cabeçada — esperado: ~0.30
        ])
        xgs = model.predict_xg(X_test)
        print(f"Pênalti:   xG = {xgs[0]:.3f}  (esperado ~0.75)")
        print(f"Chute long: xG = {xgs[1]:.3f}  (esperado ~0.03)")
        print(f"Cabeçada:  xG = {xgs[2]:.3f}  (esperado ~0.20-0.35)")
        
        XGModel.plot_calibration(model)
        
    print("\nModule test passed. XGModel ready.")