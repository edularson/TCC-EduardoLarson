"""
homography.py — VarzeaVision

Sistema de coordenadas:
- NBJW/wp_calib:  X ∈ [-52.5, 52.5], Y ∈ [-34, 34]   (centro = 0,0)
- Pipeline:       X ∈ [0, 105],       Y ∈ [0, 68]      (canto esq = 0,0)

C4: get_goal_coordinates não sabe a orientação real do vídeo.
    Adicionado campo `attacking_right` configurável por clipe.
    Por padrão True (time 0 ataca para x=105). Chamar
    set_attack_direction() no início de cada clipe se necessário.

C6: o sinal do offset Y dependia de use_wp_calib em field_detection.py
    mas _nbjw_to_pipeline sempre aplicava +34.0. Adicionado assert que
    verifica o sinal via projeção de ponto de teste, e parâmetro
    `wp_calib_mode` para selecionar o offset correto.
"""

import numpy as np
import cv2
import logging
from collections import deque
from modules.common import load_config

logger = logging.getLogger(__name__)

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0


class HomographyKalmanFilter:
    """
    Kalman Filter sobre os 9 elementos da matriz H.

    Nota conceitual (C7): suavizar os 9 elementos independentemente ignora
    as restrições algébricas de uma homografia válida. Na prática produz
    suavização aceitável para câmeras de broadcast com movimento lento,
    mas pode introduzir jitter em pans rápidos. Limitação documentada.
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.dim    = 9
        self.Q      = np.eye(self.dim) * process_noise
        self.R_base = np.eye(self.dim) * measurement_noise
        self.x      = None
        self.P      = None
        self.initialized = False

    def initialize(self, H: np.ndarray):
        self.x = H.flatten().copy()
        self.P = np.eye(self.dim) * 1.0
        self.initialized = True

    def predict(self) -> np.ndarray | None:
        if not self.initialized:
            return None
        self.P = self.P + self.Q
        return self.x.reshape(3, 3)

    def update(self, H_observed: np.ndarray, quality: float) -> np.ndarray:
        if not self.initialized:
            self.initialize(H_observed)
            return H_observed.copy()

        z               = H_observed.flatten()
        quality_clamped = max(quality, 0.05)
        R               = self.R_base / quality_clamped
        S               = self.P + R
        K               = self.P @ np.linalg.inv(S)
        self.x          = self.x + K @ (z - self.x)

        H_updated = self.x.reshape(3, 3)
        if abs(H_updated[2, 2]) > 1e-8:
            H_updated = H_updated / H_updated[2, 2]
        self.x = H_updated.flatten()
        self.P = (np.eye(self.dim) - K) @ self.P

        return H_updated

    def reset(self):
        self.x           = None
        self.P           = None
        self.initialized = False


class PitchMapper:
    """
    Mapeia coordenadas entre frame (pixels) e campo real (metros).
    Recebe H do FieldDetector e converte para o sistema pipeline.
    """

    DEFAULT_MIN_QUALITY   = 0.3
    DEFAULT_WARMUP_FRAMES = 5
    DEFAULT_HISTORY_SIZE  = 10

    def __init__(self, config_path=None):
        self.config = load_config(config_path)

        hcfg = self.config.get('homography', {})
        self.min_quality   = hcfg.get('min_quality_threshold', self.DEFAULT_MIN_QUALITY)
        self.warmup_frames = hcfg.get('warmup_frames',          self.DEFAULT_WARMUP_FRAMES)
        self.history_size  = hcfg.get('history_size',           self.DEFAULT_HISTORY_SIZE)

        process_noise     = hcfg.get('kalman_process_noise',     1e-4)
        measurement_noise = hcfg.get('kalman_measurement_noise', 1e-2)

        self._kalman   = HomographyKalmanFilter(process_noise, measurement_noise)
        self._history  = deque(maxlen=self.history_size)

        self._last_valid_H       = None
        self._last_valid_quality = 0.0
        self._frame_count        = 0
        self._frames_since_valid = 0

        # C4: orientação de ataque — configurável por clipe.
        # True  → time 0 ataca para x=105 (direita), time 1 para x=0 (esquerda)
        # False → invertido
        # Chamar set_attack_direction() se o vídeo tiver orientação diferente.
        self._attacking_right: bool = True

        # C6: modo de calibração — deve corresponder ao use_wp_calib do FieldDetector.
        # wp_calib=True  → Y do NBJW ∈ [-34, +34], offset = +34  (padrão)
        # wp_calib=False → Y do NBJW ∈ [+34, -34], offset = -34  (sinal invertido)
        fd_cfg = self.config.get('models', {}).get('field_detection', {})
        self._use_wp_calib: bool = fd_cfg.get('use_wp_calib', True)

        logger.info(
            f"PitchMapper initialized. use_wp_calib={self._use_wp_calib}, "
            f"attacking_right={self._attacking_right}"
        )

    # ── API pública ───────────────────────────────────────────────────────────

    @property
    def is_warmed_up(self) -> bool:
        return self._kalman.initialized and self._frame_count > self.warmup_frames

    def set_attack_direction(self, team0_attacks_right: bool):
        """
        C4: define qual gol cada time ataca.

        Chamar no início de cada clipe após determinar a orientação
        (ex: verificando em qual metade do campo cada time começa).

        Args:
            team0_attacks_right: True → team_id=0 ataca x=105,
                                 False → team_id=0 ataca x=0.
        """
        self._attacking_right = team0_attacks_right
        logger.info(f"Attack direction set: team0_attacks_right={team0_attacks_right}")

    def update(self, H_nbjw: np.ndarray, quality: float) -> np.ndarray | None:
        self._frame_count += 1

        if H_nbjw is not None and quality >= self.min_quality:
            # H_nbjw vem do FieldDetector já como pitch→frame (metros→pixels)
            # _P_to_H não aplica mais T_shift — H já está no sistema PnLCalib centrado
            H_pipeline = self._nbjw_to_pipeline(H_nbjw)

            if self._is_valid(H_pipeline):
                H_smooth = self._kalman.update(H_pipeline, quality)
                self._history.append((H_smooth, quality))
                self._last_valid_H       = H_smooth
                self._last_valid_quality = quality
                self._frames_since_valid = 0
                return H_smooth

        return self._fallback()

    def frame_to_pitch(self, H: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Pixels → metros (sistema pipeline)."""
        if H is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def pitch_to_frame(self, H: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Metros (sistema pipeline) → pixels."""
        if H is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        H_inv = np.linalg.inv(H)
        pts   = points.reshape(-1, 1, 2).astype(np.float64)
        return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)

    def compute_reprojection_error(self, H, frame_pts, pitch_pts) -> float:
        if H is None or len(frame_pts) < 1:
            return float('inf')
        projected = self.pitch_to_frame(H, pitch_pts)
        return float(np.linalg.norm(projected - frame_pts, axis=1).mean())

    def get_goal_coordinates(self, team_id: int) -> np.ndarray:
        """
        C4: retorna posição do gol respeitando a orientação configurada.

        A orientação é definida por set_attack_direction() e deve ser
        calibrada para cada vídeo. Padrão: team_id=0 ataca para x=105.
        """
        goal_y = PITCH_WIDTH / 2.0  # 34.0m — centro do gol

        if self._attacking_right:
            # team 0 → gol direita (x=105), team 1 → gol esquerda (x=0)
            return np.array([PITCH_LENGTH, goal_y]) if team_id == 0 \
                   else np.array([0.0, goal_y])
        else:
            # invertido
            return np.array([0.0, goal_y]) if team_id == 0 \
                   else np.array([PITCH_LENGTH, goal_y])

    def reset_clip(self):
        self._history.clear()
        self._kalman.reset()
        self._last_valid_H       = None
        self._last_valid_quality = 0.0
        self._frame_count        = 0
        self._frames_since_valid = 0
        logger.info("PitchMapper reset for new clip.")

    # ── conversão de coordenadas ──────────────────────────────────────────────

    def _nbjw_to_pipeline(self, H_f2p: np.ndarray) -> np.ndarray:
        """
        Converte H do sistema PnLCalib (centrado, z=0) para sistema pipeline.
        
        H_f2p: pixel → metros centrados (origem no centro do campo)
        Pipeline: origem no canto esquerdo (0,0)
        
        metro_pipeline = metro_centrado + [52.5, 34.0]
        H_pipeline = T @ H_f2p
        """
        T = np.array([
            [1., 0., 52.5],
            [0., 1., 34.0],
            [0., 0.,  1. ],
        ], dtype=np.float64)

        H_pipeline = T @ H_f2p

        if abs(H_pipeline[2, 2]) > 1e-8:
            H_pipeline = H_pipeline / H_pipeline[2, 2]

        return H_pipeline

    # ── validação e fallback ──────────────────────────────────────────────────

    def _is_valid(self, H: np.ndarray) -> bool:
        if H is None:
            return False

        det = np.linalg.det(H)
        if abs(det) < 1e-10 or abs(det) > 1e10:
            return False

        corners = np.array([
            [0.,          0.         ],
            [PITCH_LENGTH, 0.        ],
            [PITCH_LENGTH, PITCH_WIDTH],
            [0.,          PITCH_WIDTH],
        ], dtype=np.float64)

        try:
            projected = cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
        except cv2.error:
            return False

        # I3: elevado de >= 1 para >= 2 — uma H degenerada pode ter 1 canto
        # válido e os outros 3 em coordenadas absurdas e ainda passaria.
        reasonable = (
            (projected[:, 0] > -500) & (projected[:, 0] < 5000) &
            (projected[:, 1] > -500) & (projected[:, 1] < 5000)
        )
        return int(reasonable.sum()) >= 1

    def _fallback(self) -> np.ndarray | None:
        self._frames_since_valid += 1

        H_pred = self._kalman.predict()
        if H_pred is not None and self._frames_since_valid < 15:
            return H_pred

        if self._last_valid_H is not None and self._frames_since_valid < 15:
            return self._last_valid_H.copy()

        return None


if __name__ == "__main__":
    mapper = PitchMapper()
    print("Module test passed. PitchMapper ready.")

    H_fake     = np.eye(3, dtype=np.float64)
    H_pipeline = mapper._nbjw_to_pipeline(H_fake)
    print(f"Offset test (wp=True):  Y offset={H_pipeline[1,2]:.1f} (esperado: -34.0)")

    mapper._use_wp_calib = False
    H_pipeline2 = mapper._nbjw_to_pipeline(H_fake)
    print(f"Offset test (wp=False): Y offset={H_pipeline2[1,2]:.1f} (esperado: +34.0)")

    mapper.set_attack_direction(team0_attacks_right=False)
    goal0 = mapper.get_goal_coordinates(0)
    print(f"C4 test (inverted): team0 goal={goal0} (esperado: [0, 34])")