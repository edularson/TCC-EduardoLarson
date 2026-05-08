"""
homography.py — VarzeaVision
PitchMapper com suavização temporal, Kalman Filter e fallback inteligente.

Melhorias em relação à versão anterior:
1. Suavização temporal com média ponderada por qualidade (deque):
   frames com keypoints de boa qualidade têm mais peso na média.
2. Kalman Filter sobre os 9 elementos da matriz H:
   propaga a estimativa de H entre frames, suavizando ruído de detecção.
   Baseado em: "Video-based Sequential Bayesian Homography Estimation
   for Soccer Field Registration" (2024, ScienceDirect).
3. Fallback em cascata:
   frame atual → média ponderada → Kalman prediction → última H válida.
4. A inversão do eixo X foi centralizada aqui com documentação clara.
5. Consistência de coordenadas: todas as funções públicas trabalham no
   mesmo espaço (metros reais, eixo X já corrigido).
"""

import yaml
import numpy as np
import cv2
from collections import deque
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from pathlib import Path


class HomographyQuality:
    """Container para uma estimativa de H com seu score de qualidade."""
    def __init__(self, H: np.ndarray, quality: float):
        self.H = H.copy()
        self.quality = quality  # float [0, 1]


class HomographyKalmanFilter:
    """
    Kalman Filter simples sobre os 9 elementos da matriz de homografia.

    Estado: vetor de 9 elementos (H.flatten()).
    Modelo de processo: H muda suavemente entre frames (random walk).
    Modelo de medição: observamos H diretamente (com ruído de detecção).

    Referência: BHITK — Bayesian Homography Inference from Tracked Keypoints
    (arxiv 2311.10361, publicado ScienceDirect 2024).
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.dim = 9  # H é 3x3, 9 elementos

        # Covariância do processo (quão rápido H pode mudar entre frames)
        self.Q = np.eye(self.dim) * process_noise

        # Covariância da medição (quão confiável é a H observada)
        # Será escalada pelo inverso do quality_score no update
        self.R_base = np.eye(self.dim) * measurement_noise

        # Estado inicial e covariância de estado (não inicializados até
        # receber a primeira observação)
        self.x = None  # (9,) — H.flatten()
        self.P = None  # (9,9) — covariância do estado

        self.initialized = False

    def initialize(self, H: np.ndarray):
        """Inicializa com a primeira homografia observada."""
        self.x = H.flatten().copy()
        # Alta incerteza inicial
        self.P = np.eye(self.dim) * 1.0
        self.initialized = True

    def predict(self) -> np.ndarray:
        """
        Etapa de predição: propaga o estado sem nova observação.
        Modelo: H(t) ≈ H(t-1) (câmera tática quase estática).
        Retorna H predita como array (3,3).
        """
        if not self.initialized:
            return None
        # x_pred = x (identidade de processo)
        # P_pred = P + Q
        self.P = self.P + self.Q
        return self.x.reshape(3, 3)

    def update(self, H_observed: np.ndarray, quality: float) -> np.ndarray:
        """
        Etapa de update com nova observação.
        quality em [0,1] escala a confiança na medição:
        quality=1.0 → muito confiável, quality=0.1 → pouco confiável.

        Retorna H atualizada como array (3,3).
        """
        if not self.initialized:
            self.initialize(H_observed)
            return H_observed.copy()

        z = H_observed.flatten()

        # Escala ruído de medição pelo inverso da qualidade
        # quality baixa → R grande → Kalman confia menos na observação
        quality_clamped = max(quality, 0.05)  # evita divisão por 0
        R = self.R_base / quality_clamped

        # Ganho de Kalman: K = P * (P + R)^-1
        S = self.P + R
        K = self.P @ np.linalg.inv(S)

        # Update do estado
        self.x = self.x + K @ (z - self.x)

        # Update da covariância: P = (I - K) * P
        I = np.eye(self.dim)
        self.P = (I - K) @ self.P

        return self.x.reshape(3, 3)

    def reset(self):
        self.x = None
        self.P = None
        self.initialized = False


class PitchMapper:
    """
    Mapeia coordenadas entre frame (pixels) e campo real (metros).

    Pipeline de estimativa de H por frame:
        1. Tenta usar H do frame atual (se quality >= min_quality_threshold)
        2. Se ruim, faz weighted average com deque histórico
        3. Se ainda ruim, usa predição do Kalman Filter
        4. Se Kalman não inicializado, usa última H válida salva
        5. Se nada disponível, retorna None (pipeline não processa esse frame)
    """

    # ------------------------------------------------------------------
    # Thresholds de qualidade (podem ser sobrescritos no config.yaml)
    # ------------------------------------------------------------------
    DEFAULT_MIN_QUALITY    = 0.3   # abaixo disso, não usa H direta do frame
    DEFAULT_WARMUP_FRAMES  = 10    # acumula N frames antes de fixar H inicial
    DEFAULT_HISTORY_SIZE   = 10    # tamanho do deque de histórico

    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_pitch = SoccerPitchConfiguration()
        hcfg = self.config.get('homography', {})

        self.min_quality      = hcfg.get('min_quality_threshold',  self.DEFAULT_MIN_QUALITY)
        self.warmup_frames    = hcfg.get('warmup_frames',           self.DEFAULT_WARMUP_FRAMES)
        self.history_size     = hcfg.get('history_size',            self.DEFAULT_HISTORY_SIZE)

        # Deque de HomographyQuality para média ponderada
        self._history: deque[HomographyQuality] = deque(maxlen=self.history_size)

        # Kalman Filter sobre elementos de H
        process_noise     = hcfg.get('kalman_process_noise',     1e-4)
        measurement_noise = hcfg.get('kalman_measurement_noise', 1e-2)
        self._kalman = HomographyKalmanFilter(process_noise, measurement_noise)

        # Última H válida para fallback de último recurso
        self._last_valid_H: np.ndarray | None = None
        self._last_valid_quality: float = 0.0

        # Contador de frames processados no clip atual
        self._frame_count = 0

        # Fase de warmup: acumula frames antes de fazer Kalman update
        self._warmup_buffer: list[HomographyQuality] = []

        print("PitchMapper initialized.")
        print(f"  min_quality={self.min_quality}, warmup={self.warmup_frames}, "
              f"history={self.history_size}")

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def get_homography(self,
                       frame_pts: np.ndarray,
                       pitch_pts: np.ndarray,
                       quality: float) -> np.ndarray | None:
        """
        Retorna a melhor estimativa disponível da matriz H (3x3) para o
        frame atual, aplicando a cascata de fallbacks.

        Args:
            frame_pts : (N, 2) keypoints em pixels (do FieldDetector)
            pitch_pts : (N, 2) keypoints em metros reais
            quality   : score [0,1] do FieldDetector para este frame

        Returns:
            H (3x3) ou None se nenhuma estimativa disponível
        """
        self._frame_count += 1
        H_current = None

        # --- Tenta calcular H do frame atual ---
        if len(frame_pts) >= 4 and quality > 0.0:
            H_raw = self._compute_H_ransac(frame_pts, pitch_pts)
            if H_raw is not None:
                H_current = H_raw
                hq = HomographyQuality(H_raw, quality)

                # Fase de warmup: apenas acumula sem usar Kalman
                if self._frame_count <= self.warmup_frames:
                    self._warmup_buffer.append(hq)
                    self._history.append(hq)
                    if self._frame_count == self.warmup_frames:
                        self._finish_warmup()
                else:
                    # Atualiza Kalman com nova observação
                    self._history.append(hq)
                    if quality >= self.min_quality:
                        H_kalman = self._kalman.update(H_raw, quality)
                        self._last_valid_H = H_kalman
                        self._last_valid_quality = quality
                        return H_kalman
                    else:
                        # Qualidade baixa: Kalman faz predict (sem update)
                        H_pred = self._kalman.predict()
                        if H_pred is not None:
                            return H_pred

        # --- Fallback 1: média ponderada do histórico ---
        H_avg = self._weighted_average()
        if H_avg is not None:
            return H_avg

        # --- Fallback 2: predição do Kalman (sem nova observação) ---
        H_pred = self._kalman.predict()
        if H_pred is not None:
            return H_pred

        # --- Fallback 3: última H válida salva ---
        if self._last_valid_H is not None:
            return self._last_valid_H.copy()

        # Nenhuma estimativa disponível
        return None

    def frame_to_pitch(self, H: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Projeta pontos em pixels para coordenadas reais do campo (metros).

        NOTA SOBRE O EIXO X:
        O SoccerPitchConfiguration define x crescendo da esquerda para direita
        (0 = gol esquerdo, 120 = gol direito). Dependendo da orientação da câmera,
        pode ser necessário inverter.

        A inversão está DESABILITADA por padrão. Ative em config.yaml:
            homography:
              invert_x_axis: true

        Se você estiver vendo os jogadores do time certo no lado errado do campo,
        ative essa opção.
        """
        if H is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)

        pts = points.reshape(-1, 1, 2).astype(np.float64)
        result = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

        if self.config.get('homography', {}).get('invert_x_axis', False):
            pitch_length = np.array(self.config_pitch.vertices)[:, 0].max() / 100.0
            result[:, 0] = pitch_length - result[:, 0]

        return result

    def pitch_to_frame(self, H: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Projeta pontos do campo (metros) para pixels no frame."""
        if H is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)

        H_inv = np.linalg.inv(H)
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)

    def compute_reprojection_error(self,
                                   H: np.ndarray,
                                   frame_pts: np.ndarray,
                                   pitch_pts: np.ndarray) -> float:
        """
        Calcula o reprojection error médio em pixels para um H dado.
        Útil para logging e debug.
        """
        if H is None or len(frame_pts) < 1:
            return float('inf')

        projected = self.pitch_to_frame(H, pitch_pts)
        errors = np.linalg.norm(projected - frame_pts, axis=1)
        return float(errors.mean())

    def get_goal_coordinates(self, team_id: int) -> np.ndarray:
        """
        Retorna o centro do gol adversário em metros reais.

        IMPORTANTE: estas coordenadas são no mesmo espaço que frame_to_pitch()
        retorna — se invert_x_axis=True, elas também são invertidas aqui.
        """
        pitch_length = np.array(self.config_pitch.vertices)[:, 0].max() / 100.0
        pitch_width  = np.array(self.config_pitch.vertices)[:, 1].max() / 100.0
        goal_y = pitch_width / 2.0  # centro do campo em Y

        if self.config.get('homography', {}).get('invert_x_axis', False):
            # Com inversão: time 0 ataca para x=0, time 1 para x=pitch_length
            if team_id == 0:
                return np.array([0.0, goal_y])
            else:
                return np.array([pitch_length, goal_y])
        else:
            # Sem inversão (padrão): time 0 ataca para x=pitch_length
            if team_id == 0:
                return np.array([pitch_length, goal_y])
            else:
                return np.array([0.0, goal_y])

    def reset_clip(self):
        """Reseta estado entre clips. Chame ao iniciar processamento de novo vídeo."""
        self._history.clear()
        self._warmup_buffer.clear()
        self._kalman.reset()
        self._last_valid_H = None
        self._last_valid_quality = 0.0
        self._frame_count = 0
        print("PitchMapper reset for new clip.")

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _compute_H_ransac(self,
                          frame_pts: np.ndarray,
                          pitch_pts: np.ndarray) -> np.ndarray | None:
        """
        Calcula H via cv2.findHomography com RANSAC.
        Mapeia pitch_pts → frame_pts (para posterior inversão com perspectiveTransform).

        O RANSAC filtra outliers automaticamente, tornando o cálculo muito
        mais robusto que cv2.getPerspectiveTransform (que usa exatamente 4 pontos).
        """
        if len(frame_pts) < 4:
            return None
        try:
            # Direção: frame → pitch (para usar em perspectiveTransform no frame_to_pitch)
            H, mask = cv2.findHomography(
                frame_pts.reshape(-1, 1, 2),
                pitch_pts.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,  # pixels
                confidence=0.995,
                maxIters=2000
            )
            if H is None:
                return None
            inlier_count = mask.sum() if mask is not None else 0
            if inlier_count < 4:
                return None
            return H
        except cv2.error:
            return None

    def _finish_warmup(self):
        """
        Ao fim do warmup, inicializa o Kalman com a média ponderada dos
        frames acumulados para ter um estado inicial mais robusto.
        """
        H_init = self._weighted_average_from(self._warmup_buffer)
        if H_init is not None:
            self._kalman.initialize(H_init)
            self._last_valid_H = H_init.copy()
            self._last_valid_quality = np.mean([hq.quality for hq in self._warmup_buffer])
            print(f"[PitchMapper] Warmup done. Kalman initialized with "
                  f"{len(self._warmup_buffer)} frames, "
                  f"avg quality={self._last_valid_quality:.2f}")
        self._warmup_buffer.clear()

    def _weighted_average(self) -> np.ndarray | None:
        """Média ponderada por qualidade sobre o deque histórico."""
        return self._weighted_average_from(list(self._history))

    def _weighted_average_from(self, items: list) -> np.ndarray | None:
        """Média ponderada por qualidade sobre uma lista de HomographyQuality."""
        if not items:
            return None

        good = [hq for hq in items if hq.quality >= self.min_quality]
        if len(good) < 1:
            return None

        weights = np.array([hq.quality for hq in good])
        weights /= weights.sum()

        H_avg = np.zeros((3, 3), dtype=np.float64)
        for hq, w in zip(good, weights):
            H_avg += w * hq.H

        return H_avg


# ------------------------------------------------------------------
# Alias de compatibilidade com o código antigo
# (para não quebrar main.py que chama compute_homography_from_points)
# ------------------------------------------------------------------

    def compute_homography_from_points(self,
                                       frame_points: np.ndarray,
                                       pitch_points: np.ndarray) -> 'ViewTransformer':
        """
        Compatibilidade com a API antiga do main.py.
        Retorna um ViewTransformer, igual antes.
        ATENÇÃO: prefira usar get_homography() + frame_to_pitch() diretamente.
        """
        H = self._compute_H_ransac(frame_points, pitch_points)
        if H is None:
            raise ValueError(f"Could not compute homography from {len(frame_points)} points")
        transformer = ViewTransformer(source=frame_points, target=pitch_points)
        return transformer


if __name__ == "__main__":
    mapper = PitchMapper()
    print("Module test passed. PitchMapper ready.")

    # Teste rápido do Kalman Filter
    kf = HomographyKalmanFilter()
    H_test = np.eye(3, dtype=np.float64)
    H_test[0, 2] = 50.0  # translação de teste
    H_out = kf.update(H_test, quality=0.8)
    print(f"Kalman test: input H[0,2]={H_test[0,2]:.1f}, output H[0,2]={H_out[0,2]:.3f}")
    H_pred = kf.predict()
    print(f"Kalman predict: H[0,2]={H_pred[0,2]:.3f}")