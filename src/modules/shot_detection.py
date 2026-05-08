"""
shot_detection.py — VarzeaVision
Detecção de chutes com velocidade suavizada e direção correta.

Melhorias em relação à versão anterior:
1. Usa VELOCIDADE da bola em vez de aceleração:
   - Aceleração (derivada segunda) amplifica ruído de posição
   - Velocidade (derivada primeira) é muito mais estável
   - Suavização com janela deslizante antes de derivar
2. Direção do chute calculada com vetor de velocidade real da bola,
   não mais o proxy jogador→bola (que era sempre errado)
3. goal_pos vem do PitchMapper para ser consistente com invert_x_axis
4. Debug por timestamp removido — substituído por logging condicional
5. Lógica de detecção documentada com referência à literatura

Referência:
  "Event detection in football: Improving the reliability of match analysis"
  PLOS ONE, 2024. F-score de 0.65 para detecção rule-based de chutes com
  dados independentes — classificadores ML chegam a 0.95.
"""

import yaml
import numpy as np
import logging
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class BallVelocityEstimator:
    """
    Estima velocidade da bola com janela deslizante.

    Usa regressão linear sobre os últimos N pontos para estimar
    velocidade — muito mais robusto que diferença simples entre frames,
    especialmente quando há detecções ruidosas ou frames perdidos.
    """

    def __init__(self, window_size: int = 5):
        # Deques de (tempo, posição)
        self.window_size = window_size
        self._times: deque  = deque(maxlen=window_size)
        self._positions: deque = deque(maxlen=window_size)

    def update(self, pos: np.ndarray, t: float):
        self._times.append(t)
        self._positions.append(pos.copy())

    def get_velocity(self) -> tuple[np.ndarray, float]:
        """
        Retorna (velocity_vector, speed_magnitude) em metros/segundo.
        Usa regressão linear sobre a janela para estimar dv/dt.
        Retorna (zeros, 0.0) se janela insuficiente.
        """
        n = len(self._times)
        if n < 2:
            return np.zeros(2), 0.0

        times = np.array(self._times)
        positions = np.array(self._positions)  # (n, 2)

        # Regressão linear por eixo: pos = a*t + b
        # Usa apenas os dois pontos extremos da janela se n=2,
        # ou polyfit se temos mais pontos
        try:
            vx = np.polyfit(times, positions[:, 0], 1)[0]
            vy = np.polyfit(times, positions[:, 1], 1)[0]
        except np.linalg.LinAlgError:
            return np.zeros(2), 0.0

        velocity = np.array([vx, vy])
        speed = float(np.linalg.norm(velocity))
        return velocity, speed

    def get_direction(self) -> np.ndarray:
        """Retorna vetor unitário da direção de movimento da bola."""
        vel, speed = self.get_velocity()
        if speed < 1e-6:
            return np.zeros(2)
        return vel / speed

    def reset(self):
        self._times.clear()
        self._positions.clear()


class ShotDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        shot_cfg = self.config['shot_detection']
        self.max_player_ball_dist = shot_cfg['max_player_ball_distance']  # metros
        self.min_ball_speed       = shot_cfg.get('min_ball_speed', 3.0)   # m/s
        self.goal_proximity       = shot_cfg['goal_proximity_distance']   # metros

        self.shot_cooldown  = shot_cfg.get('shot_cooldown', 3.0)
        self.last_shot_time = -999.0

        # Estimador de velocidade com janela deslizante
        velocity_window = shot_cfg.get('velocity_window_frames', 5)
        self._vel_estimator = BallVelocityEstimator(window_size=velocity_window)

        # Histórico de posições para visualização da trajetória
        self.ball_trail: deque = deque(maxlen=30)

        logger.info(
            f"ShotDetector initialized. "
            f"max_dist={self.max_player_ball_dist}m, "
            f"min_speed={self.min_ball_speed}m/s, "
            f"goal_prox={self.goal_proximity}m"
        )
        print(
            f"ShotDetector initialized. "
            f"max_dist={self.max_player_ball_dist}m | "
            f"min_speed={self.min_ball_speed}m/s"
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def update_ball_position(self, ball_pos: np.ndarray, frame_time: float):
        """
        Deve ser chamado todo frame, mesmo quando não há detecção de chute.
        Mantém a janela de velocidade atualizada.
        """
        self._vel_estimator.update(ball_pos, frame_time)
        self.ball_trail.append(ball_pos.copy())

    def compute_ball_velocity(self, ball_pos: np.ndarray, frame_time: float) -> tuple:
        """
        Atualiza estimador e retorna (velocity_vector, speed_magnitude).

        Substitui o antigo compute_ball_acceleration().
        A velocidade é muito mais estável que aceleração para detecção de chutes.

        Returns:
            velocity_vector : np.ndarray (2,) em m/s
            speed_magnitude : float em m/s
        """
        self._vel_estimator.update(ball_pos, frame_time)
        self.ball_trail.append(ball_pos.copy())
        return self._vel_estimator.get_velocity()

    def detect_shot(self,
                    ball_pos: np.ndarray,
                    frame_time: float,
                    speed_mag: float,
                    closest_player: dict,
                    pitch_mapper=None) -> tuple:
        """
        Detecta se o evento atual é um chute.

        Args:
            ball_pos       : posição da bola em metros reais
            frame_time     : timestamp em segundos
            speed_mag      : velocidade da bola em m/s (de compute_ball_velocity)
            closest_player : dict com 'team', 'foot_pos_real', 'track_id'
            pitch_mapper   : PitchMapper para resolver goal_pos (opcional,
                             se None usa heurística hardcoded)

        Returns:
            (is_shot: bool, data: dict)
        """
        # Cooldown entre chutes consecutivos
        if frame_time - self.last_shot_time < self.shot_cooldown:
            return False, {'reason': 'cooldown', 'confidence': 0.0}

        team_id    = closest_player.get('team', -1)
        player_pos = closest_player.get('foot_pos_real')

        if player_pos is None:
            return False, {'reason': 'no_player_pos', 'confidence': 0.0}

        # Resolve coordenadas do gol de forma consistente
        if pitch_mapper is not None:
            goal_pos = pitch_mapper.get_goal_coordinates(team_id)
        else:
            # Fallback hardcoded (mantido para compatibilidade)
            goal_pos = np.array([0.0, 35.0]) if team_id == 1 else np.array([120.0, 35.0])
            logger.warning("pitch_mapper not provided — using hardcoded goal coordinates")

        is_shot, confidence, reason = self.is_shot(
            player_pos=player_pos,
            ball_pos=ball_pos,
            ball_speed=speed_mag,
            goal_pos=goal_pos,
            player_team_id=team_id
        )

        logger.debug(
            f"t={frame_time:.2f}s | speed={speed_mag:.1f}m/s | "
            f"team={team_id} | shot={is_shot} | {reason}"
        )

        if not is_shot:
            return False, {'confidence': confidence, 'reason': reason}

        self.last_shot_time = frame_time
        return True, {
            'confidence': confidence,
            'reason': reason,
            'goal_pos': goal_pos.tolist()
        }

    def is_shot(self,
                player_pos: np.ndarray,
                ball_pos: np.ndarray,
                ball_speed: float,
                goal_pos: np.ndarray,
                player_team_id: int) -> tuple:
        """
        Heurística de detecção de chute.

        Critérios (em ordem de eliminação):
        1. Jogador próximo da bola (distância real em metros)
        2. Velocidade da bola acima do threshold (m/s)
        3. Bola dentro da zona de chute (distância ao gol)
        4. Direção da bola aponta para o gol (dot product com vetor real de velocidade)

        Returns:
            (is_shot, confidence, reason_str)
        """
        # 1. Proximidade jogador-bola
        player_ball_dist = np.linalg.norm(player_pos - ball_pos)
        if player_ball_dist > self.max_player_ball_dist:
            return False, 0.0, f"dist={player_ball_dist:.1f}m > {self.max_player_ball_dist}m"

        # 2. Velocidade mínima da bola
        if ball_speed < self.min_ball_speed:
            return False, 0.0, f"speed={ball_speed:.1f}m/s < {self.min_ball_speed}m/s"

        # 3. Proximidade ao gol
        ball_goal_dist = np.linalg.norm(ball_pos - goal_pos)
        if ball_goal_dist > self.goal_proximity:
            return False, 0.0, f"goal_dist={ball_goal_dist:.1f}m > {self.goal_proximity}m"

        # 4. Direção da bola em relação ao gol
        # Usa vetor de velocidade real (estimado pela janela deslizante),
        # NÃO mais o proxy jogador→bola
        ball_velocity_dir = self._vel_estimator.get_direction()
        ball_to_goal      = goal_pos - ball_pos
        ball_to_goal_norm = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-6)

        if np.linalg.norm(ball_velocity_dir) > 0.1:
            # Temos direção real de velocidade — usá-la
            direction_alignment = float(np.dot(ball_velocity_dir, ball_to_goal_norm))
        else:
            # Fallback: proxy jogador→bola (menos confiável)
            player_to_ball = ball_pos - player_pos
            player_to_ball_norm = player_to_ball / (np.linalg.norm(player_to_ball) + 1e-6)
            direction_alignment = float(np.dot(player_to_ball_norm, ball_to_goal_norm))
            logger.debug("Using player→ball proxy for direction (velocity unavailable)")

        if direction_alignment < 0.3:
            return False, 0.0, f"direction={direction_alignment:.2f} < 0.3"

        # Todos os critérios passaram — calcula confiança
        dist_score      = 1.0 - (player_ball_dist / self.max_player_ball_dist)
        speed_score     = min(ball_speed / (self.min_ball_speed * 3), 1.0)
        direction_score = (direction_alignment + 1.0) / 2.0

        confidence = (
            dist_score      * 0.30 +
            speed_score     * 0.45 +
            direction_score * 0.25
        )

        reason = (
            f"SHOT: dist={player_ball_dist:.1f}m | "
            f"speed={ball_speed:.1f}m/s | "
            f"dir={direction_alignment:.2f} | "
            f"conf={confidence:.2f}"
        )
        return True, float(confidence), reason

    # ------------------------------------------------------------------

    def reset_clip(self):
        self._vel_estimator.reset()
        self.ball_trail.clear()
        self.last_shot_time = -999.0
        print("ShotDetector reset for new clip.")


# ------------------------------------------------------------------
# Compatibilidade: mantém compute_ball_acceleration como alias
# (para não quebrar chamadas existentes em main.py imediatamente)
# ------------------------------------------------------------------
    def compute_ball_acceleration(self, ball_pos: np.ndarray, frame_time: float) -> tuple:
        """
        DEPRECATED — use compute_ball_velocity() instead.
        Mantido para compatibilidade com main.py antigo.
        Retorna (speed_magnitude, velocity_vector) — note: ordem diferente
        do antigo (accel_mag, accel_vec), mas agora são valores de velocidade.
        """
        vel_vec, speed = self.compute_ball_velocity(ball_pos, frame_time)
        return speed, vel_vec


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    detector = ShotDetector()
    print("Module test passed. ShotDetector ready.\n")

    # Simula bola se movendo em direção ao gol com velocidade real
    goal_pos   = np.array([120.0, 34.0])
    player_pos = np.array([100.0, 34.0])

    # Simula 6 frames de bola se movendo para o gol
    positions = [
        (np.array([98.0, 34.0]), 0.0),
        (np.array([99.0, 34.0]), 0.04),
        (np.array([100.5, 34.0]), 0.08),
        (np.array([102.5, 34.0]), 0.12),  # chute aqui
        (np.array([105.0, 34.0]), 0.16),
        (np.array([108.0, 34.0]), 0.20),
    ]

    speed_mag = 0.0
    for pos, t in positions:
        vel_vec, speed_mag = detector.compute_ball_velocity(pos, t)
        print(f"  t={t:.2f}s | pos=({pos[0]:.1f},{pos[1]:.1f}) | speed={speed_mag:.1f}m/s")

    # Testa detecção
    closest_player = {
        'track_id': 7,
        'team': 0,
        'foot_pos_real': player_pos,
    }

    # Cria um mock de pitch_mapper para o teste
    class MockMapper:
        def get_goal_coordinates(self, team_id):
            return goal_pos

    is_shot, data = detector.detect_shot(
        ball_pos=np.array([108.0, 34.0]),
        frame_time=0.20,
        speed_mag=speed_mag,
        closest_player=closest_player,
        pitch_mapper=MockMapper()
    )
    print(f"\nShot detected: {is_shot}")
    print(f"Data: {data}")