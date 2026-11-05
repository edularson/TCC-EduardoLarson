import torch
import numpy as np
import supervision as sv
import cv2
from modules.common import load_config


class FootballTracker:
    """
    Tracker de jogadores, goleiros e árbitros.
    A bola NÃO entra neste tracker — use BallKalmanTracker separadamente.
    """

    def __init__(self, config_path=None):
        self.config = load_config(config_path)

        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = 'cpu'

        try:
            self.tracker = sv.BotSort(
                reid_weights=None,
                device=self.device,
                half=False,
                max_age=self.config['tracking']['max_age'],
                min_hits=self.config['tracking']['min_hits'],
            )
            self.tracker_type = 'botsort'
        except AttributeError:
            print("BotSORT not available. Falling back to ByteTrack.")
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=self.config['tracking']['max_age'],
                minimum_matching_threshold=0.8,
            )
            self.tracker_type = 'bytetrack'

        print(f"FootballTracker ({self.tracker_type}) initialized on device: {self.device}")

    def update(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        if self.tracker_type == 'botsort':
            return self.tracker.update_with_detections(detections=detections, frame=frame)
        else:
            return self.tracker.update_with_detections(detections=detections)

    def reset(self):
        self.tracker.reset()
        print("Tracker reset for new clip.")


class BallKalmanTracker:
    """
    Tracker da bola com Kalman Filter de 4 estados [x, y, vx, vy].

    Motivação: o BallTracker simples baseado em distância euclidiana falha
    quando o YOLO detecta marcações do campo (marca do pênalti, pequena área)
    como bola — a detecção "pula" entre a bola real e a marcação estática.

    O Kalman Filter resolve isso de duas formas:
    1. Prediz onde a bola deve estar no próximo frame baseado na velocidade atual.
    2. Rejeita detecções muito longe da predição (gate de Mahalanobis).
    Marcações estáticas têm posição fixa — depois de alguns frames o filtro
    aprende a trajetória real e rejeita saltos para posições estáticas.

    Estados:  [x, y, vx, vy]  (pixels)
    Medições: [x, y]          (pixels, centro da bbox)
    """

    # Velocidade máxima física de uma bola de futebol em pixels/frame.
    # A ~30fps, 35 m/s ≈ ~100px/frame em broadcast 1080p típico.
    # Usado para inicializar o gate de rejeição.
    MAX_SPEED_PX_PER_FRAME = 120.0

    def __init__(self,
                 max_missing_frames: int = 8,
                 process_noise: float = 10.0,
                 measurement_noise: float = 25.0,
                 gate_sigma: float = 3.0):
        """
        Args:
            max_missing_frames: frames sem detecção aceitável antes de resetar.
            process_noise: incerteza do modelo de movimento (Q). Maior = mais
                           confiança na medição, menor = mais confiança na predição.
            measurement_noise: incerteza do detector YOLO (R). Maior = filtra
                               mais agressivamente detecções ruidosas.
            gate_sigma: número de desvios padrão para gate de Mahalanobis.
                        Detecções fora do gate são rejeitadas como falsos positivos.
        """
        self.max_missing       = max_missing_frames
        self.gate_sigma        = gate_sigma
        self._missing          = 0
        self._initialized      = False
        self._last_valid_bbox  = None  # (x1,y1,x2,y2,conf) — para fallback

        # ── Matrizes do Kalman Filter ─────────────────────────────────────────
        # Estado: [x, y, vx, vy]
        self._x = np.zeros(4)          # vetor de estado
        self._P = np.eye(4) * 500.0    # covariância inicial (alta incerteza)

        # Modelo de transição: x_{k+1} = F @ x_k
        # Posição = posição anterior + velocidade * dt (dt=1 frame)
        self._F = np.array([
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])

        # Matriz de observação: z = H @ x (observamos só x,y)
        self._H = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
        ])

        # Ruído do processo (incerteza do modelo de movimento)
        self._Q = np.eye(4) * process_noise

        # Ruído de medição (incerteza do detector)
        self._R = np.eye(2) * measurement_noise

    def _predict(self):
        """Passo de predição do Kalman."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def _update(self, z: np.ndarray):
        """Passo de atualização do Kalman com medição z=[x,y]."""
        y = z - self._H @ self._x                          # inovação
        S = self._H @ self._P @ self._H.T + self._R        # covariância da inovação
        K = self._P @ self._H.T @ np.linalg.inv(S)         # ganho de Kalman
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P

    def _mahalanobis_dist(self, z: np.ndarray) -> float:
        """Distância de Mahalanobis entre medição z e predição atual."""
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        try:
            return float(np.sqrt(y @ np.linalg.inv(S) @ y))
        except np.linalg.LinAlgError:
            return float('inf')

    def _bbox_center(self, bbox: np.ndarray) -> np.ndarray:
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        ])

    def _optical_flow_predict(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray | None:
        """
        Usa Lucas-Kanade Optical Flow para estimar posição da bola
        quando o detector falha. Rastreia o patch centrado na última
        posição conhecida.
        """
        if self._last_valid_bbox is None:
            return None

        cx = (self._last_valid_bbox[0] + self._last_valid_bbox[2]) / 2
        cy = (self._last_valid_bbox[1] + self._last_valid_bbox[3]) / 2
        pt = np.array([[[cx, cy]]], dtype=np.float32)

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pt, None, **lk_params
        )

        if status is None or status[0][0] == 0:
            return None

        nx, ny = next_pt[0][0]

        # Rejeita se o deslocamento for fisicamente impossível
        dx = nx - cx
        dy = ny - cy
        if np.sqrt(dx**2 + dy**2) > self.MAX_SPEED_PX_PER_FRAME:
            return None

        print(f"[OF] Optical Flow ativo - {self._missing} frames sem detecção")
        return np.array([nx, ny])


    def update(self, ball_detections: np.ndarray,
            curr_frame: np.ndarray = None,
            prev_frame: np.ndarray = None) -> np.ndarray | None:
        """
        Args:
            ball_detections: array (N, 5) [x1,y1,x2,y2,conf] ou (0,5) vazio.
            curr_frame: frame atual (BGR numpy array).
            prev_frame: frame anterior (BGR numpy array).
        Returns:
            Array (5,) melhor detecção filtrada, ou None se não há estimativa.
        """

        # ── 1. FILTRO FÍSICO DE ÁREA E PROPORÇÃO (Hard Negative Rejection) ──
        valid_detections = []
        for det in ball_detections:
            w = det[2] - det[0]
            h = det[3] - det[1]
            area = w * h

            # Evita divisão por zero
            if h == 0 or w == 0:
                continue

            aspect_ratio = max(w/h, h/w)
            print(f"DEBUG BOLA: area={area:.1f}, aspect_ratio={aspect_ratio:.2f}, conf={det[4]:.2f}")

            # Restrições Científicas:
            if area > 350 or aspect_ratio > 2.0:
                # Rejeita o falso positivo silenciosamente
                continue

            valid_detections.append(det)

        ball_detections = np.array(valid_detections) if len(valid_detections) > 0 else np.empty((0, 5))

        # ── sem detecções ────────────────────────────────────────────────────
        if len(ball_detections) == 0:
            self._missing += 1
            if not self._initialized:
                return None
            if self._missing > self.max_missing:
                self._initialized     = False
                self._last_valid_bbox = None
                return None

            # Tenta Optical Flow antes de cair na predição pura do Kalman
            if (curr_frame is not None and prev_frame is not None
                    and self._last_valid_bbox is not None):
                of_pos = self._optical_flow_predict(prev_frame, curr_frame)
                if of_pos is not None:
                    # Atualiza Kalman com posição estimada pelo OF
                    self._predict()
                    self._update(of_pos)
                    # Reconstrói bbox sintética centrada na posição OF
                    bbox = self._last_valid_bbox.copy()
                    w = (bbox[2] - bbox[0]) / 2
                    h = (bbox[3] - bbox[1]) / 2
                    synthetic_bbox = np.array([
                        of_pos[0] - w, of_pos[1] - h,
                        of_pos[0] + w, of_pos[1] + h,
                        bbox[4],
                    ])
                    self._last_valid_bbox = synthetic_bbox
                    return synthetic_bbox

            # Fallback: predição pura do Kalman
            self._predict()
            return self._last_valid_bbox

        # ── seleciona melhor detecção por confiança ───────────────────────────
        best_idx = int(np.argmax(ball_detections[:, 4]))
        best     = ball_detections[best_idx]
        z        = self._bbox_center(best)

        # ── inicialização ─────────────────────────────────────────────────────
        if not self._initialized:
            self._x = np.array([z[0], z[1], 0., 0.])
            self._P = np.eye(4) * 500.0
            self._initialized     = True
            self._missing         = 0
            self._last_valid_bbox = best
            return best

        # ── gate de Mahalanobis ───────────────────────────────────────────────
        self._predict()
        mdist = self._mahalanobis_dist(z)

        if mdist > self.gate_sigma:
            accepted = None
            for i, det in enumerate(ball_detections):
                if i == best_idx:
                    continue
                zi = self._bbox_center(det)
                if self._mahalanobis_dist(zi) <= self.gate_sigma:
                    accepted = det
                    z        = zi
                    break

            if accepted is None:
                self._missing += 1
                if self._missing > self.max_missing:
                    self._initialized     = False
                    self._last_valid_bbox = None
                    return None
                return self._last_valid_bbox

            best = accepted

        if self._is_static(best):
            self._missing += 1
            if self._missing > self.max_missing:
                self._initialized     = False
                self._last_valid_bbox = None
                return None
            return self._last_valid_bbox

        # ── atualização ───────────────────────────────────────────────────────
        self._update(z)
        self._missing         = 0
        self._last_valid_bbox = best
        self._static_history  = []
        return best

    def reset(self):
        self._x               = np.zeros(4)
        self._P               = np.eye(4) * 500.0
        self._initialized     = False
        self._missing         = 0
        self._last_valid_bbox = None
        self._static_history = []

    def _is_static(self, bbox: np.ndarray, threshold_px: float = 5.0, min_frames: int = 8) -> bool:
        """
        Retorna True se a bola ficou parada por min_frames consecutivos.
        Detecções estáticas são provavelmente falsos positivos.
        """
        if not hasattr(self, '_static_history'):
            self._static_history = []

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        self._static_history.append(np.array([cx, cy]))
        if len(self._static_history) > min_frames:
            self._static_history.pop(0)

        if len(self._static_history) < min_frames:
            return False

        positions = np.array(self._static_history)
        max_displacement = np.max(np.linalg.norm(positions - positions[0], axis=1))
        return bool(max_displacement < threshold_px)


if __name__ == "__main__":
    tracker      = FootballTracker()
    ball_tracker = BallKalmanTracker()
    print("Module test passed. FootballTracker and BallKalmanTracker ready.")
