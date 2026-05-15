import numpy as np
import logging
from collections import deque
from modules.common import load_config

logger = logging.getLogger(__name__)

# Opção 3: cap de velocidade física.
# Velocidade máxima registrada em futebol: ~70 m/s (Roberto Carlos, chutes
# de potência extrema). Cap em 40 m/s (~144 km/h) cobre todos os chutes
# realistas e elimina velocidades artificiais causadas por saltos de detecção.
MAX_PHYSICAL_SPEED_MS = 40.0


class BallVelocityEstimator:
    """
    Estima velocidade da bola com janela deslizante via OLS vetorizado.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._times: deque     = deque(maxlen=window_size)
        self._positions: deque = deque(maxlen=window_size)
        self._last_t: float    = -1.0

    def update(self, pos: np.ndarray, t: float):
        # Gap temporal > 1s indica bola perdida — reseta janela para evitar
        # velocidade artificial (ex: 20m em 0.5s = 40 m/s falso).
        if self._last_t >= 0 and (t - self._last_t) > 1.0:
            logger.debug(f"Gap temporal {t - self._last_t:.2f}s — janela OLS resetada.")
            self._times.clear()
            self._positions.clear()

        self._last_t = t
        self._times.append(t)
        self._positions.append(pos.copy())

    def get_velocity(self) -> tuple[np.ndarray, float]:
        """Retorna (velocity_vector, speed_magnitude) em metros/segundo."""
        n = len(self._times)
        if n < 2:
            return np.zeros(2), 0.0

        times     = np.array(self._times)
        positions = np.array(self._positions)

        t_mean = np.mean(times)
        p_mean = np.mean(positions, axis=0)
        t_diff = times - t_mean
        p_diff = positions - p_mean
        var_t  = np.sum(t_diff ** 2)

        if var_t < 1e-6:
            return np.zeros(2), 0.0

        velocity = np.dot(t_diff, p_diff) / var_t
        speed    = float(np.linalg.norm(velocity))

        # Opção 3: cap de velocidade física.
        # Velocidades acima de MAX_PHYSICAL_SPEED_MS indicam salto de detecção
        # (bola real → marcação do campo ou vice-versa), não movimento real.
        if speed > MAX_PHYSICAL_SPEED_MS:
            logger.debug(
                f"Velocidade {speed:.1f} m/s acima do cap físico "
                f"({MAX_PHYSICAL_SPEED_MS} m/s) — descartada como ruído."
            )
            return np.zeros(2), 0.0

        return velocity, speed

    def get_direction(self) -> np.ndarray:
        vel, speed = self.get_velocity()
        if speed < 1e-6:
            return np.zeros(2)
        return vel / speed

    def reset(self):
        self._times.clear()
        self._positions.clear()
        self._last_t = -1.0


class ShotDetector:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)

        shot_cfg = self.config['shot_detection']
        self.max_player_ball_dist = shot_cfg['max_player_ball_distance']
        self.min_ball_speed       = shot_cfg.get('min_ball_speed', 5.0)
        self.goal_proximity       = shot_cfg['goal_proximity_distance']
        self.shot_cooldown        = shot_cfg.get('shot_cooldown', 3.0)
        self.last_shot_time       = -999.0

        velocity_window     = shot_cfg.get('velocity_window_frames', 5)
        self._vel_estimator = BallVelocityEstimator(window_size=velocity_window)
        self.ball_trail: deque = deque(maxlen=30)

        logger.info(
            f"ShotDetector initialized. max_dist={self.max_player_ball_dist}m, "
            f"min_speed={self.min_ball_speed}m/s, goal_prox={self.goal_proximity}m, "
            f"speed_cap={MAX_PHYSICAL_SPEED_MS}m/s"
        )

    def compute_ball_velocity(self, ball_pos: np.ndarray, frame_time: float) -> tuple:
        self._vel_estimator.update(ball_pos, frame_time)
        self.ball_trail.append(ball_pos.copy())
        return self._vel_estimator.get_velocity()

    def detect_shot(self,
                    ball_pos: np.ndarray,
                    frame_time: float,
                    speed_mag: float,
                    closest_player: dict,
                    pitch_mapper=None) -> tuple:

        if frame_time - self.last_shot_time < self.shot_cooldown:
            return False, {'reason': 'cooldown', 'confidence': 0.0}

        team_id    = closest_player.get('team', -1)
        player_pos = closest_player.get('foot_pos_real')

        if player_pos is None:
            return False, {'reason': 'no_player_pos', 'confidence': 0.0}

        if pitch_mapper is not None:
            goal_pos = pitch_mapper.get_goal_coordinates(team_id)
        else:
            goal_pos = np.array([0.0, 34.0]) if team_id == 1 else np.array([105.0, 34.0])

        is_shot, confidence, reason = self.is_shot(
            player_pos=player_pos,
            ball_pos=ball_pos,
            ball_speed=speed_mag,
            goal_pos=goal_pos,
        )

        if not is_shot:
            return False, {'confidence': confidence, 'reason': reason}

        self.last_shot_time = frame_time
        return True, {'confidence': confidence, 'reason': reason, 'goal_pos': goal_pos.tolist()}

    def is_shot(self,
                player_pos: np.ndarray,
                ball_pos: np.ndarray,
                ball_speed: float,
                goal_pos: np.ndarray) -> tuple:

        player_ball_dist = np.linalg.norm(player_pos - ball_pos)
        if player_ball_dist > self.max_player_ball_dist:
            return False, 0.0, f"dist={player_ball_dist:.1f}m > {self.max_player_ball_dist}m"

        if ball_speed < self.min_ball_speed:
            return False, 0.0, f"speed={ball_speed:.1f}m/s < {self.min_ball_speed}m/s"

        ball_goal_dist = np.linalg.norm(ball_pos - goal_pos)
        if ball_goal_dist > self.goal_proximity:
            return False, 0.0, f"goal_dist={ball_goal_dist:.1f}m > {self.goal_proximity}m"

        ball_velocity_dir = self._vel_estimator.get_direction()
        ball_to_goal      = goal_pos - ball_pos
        norm_btg          = np.linalg.norm(ball_to_goal)
        ball_to_goal_norm = ball_to_goal / norm_btg if norm_btg > 1e-6 else np.zeros(2)

        if np.linalg.norm(ball_velocity_dir) > 0.1:
            direction_alignment = float(np.dot(ball_velocity_dir, ball_to_goal_norm))
        else:
            player_to_ball      = ball_pos - player_pos
            norm_ptb            = np.linalg.norm(player_to_ball)
            player_to_ball_norm = player_to_ball / norm_ptb if norm_ptb > 1e-6 else np.zeros(2)
            direction_alignment = float(np.dot(player_to_ball_norm, ball_to_goal_norm))

        if direction_alignment < 0.3:
            return False, 0.0, f"direction={direction_alignment:.2f} < 0.3"

        dist_score      = max(0.0, 1.0 - (player_ball_dist / self.max_player_ball_dist))
        speed_score     = min(ball_speed / (self.min_ball_speed * 3), 1.0)
        direction_score = (direction_alignment + 1.0) / 2.0
        confidence      = (dist_score * 0.30) + (speed_score * 0.45) + (direction_score * 0.25)

        reason = (
            f"SHOT: dist={player_ball_dist:.1f}m | speed={ball_speed:.1f}m/s | "
            f"dir={direction_alignment:.2f} | conf={confidence:.2f}"
        )
        return True, float(confidence), reason

    def reset_clip(self):
        self._vel_estimator.reset()
        self.ball_trail.clear()
        self.last_shot_time = -999.0


if __name__ == "__main__":
    detector = ShotDetector()
    print("Module test passed. ShotDetector ready.")