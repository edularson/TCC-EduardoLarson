import yaml
import numpy as np
from pathlib import Path

class ShotDetector:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = Path(__file__).parent
            project_root = module_dir.parent
            config_path = project_root / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load thresholds from config
        shot_cfg = self.config['shot_detection']
        self.max_player_ball_dist = shot_cfg['max_player_ball_distance']  # meters
        self.min_ball_accel = shot_cfg['min_ball_acceleration']  # m/s²
        self.goal_proximity = shot_cfg['goal_proximity_distance']  # meters
        
        # State for tracking ball movement between frames
        self.prev_ball_pos = None  # Real-world (x,y) meters
        self.prev_frame_time = None  # Timestamp for acceleration calc
        self.ball_trail = []  # Last 5 positions for acceleration smoothing
        self.trail_length = 5
        
        self.last_shot_time = -999  # timestamp do último chute detectado
        self.shot_cooldown = 3.0    # segundos mínimos entre chutes
        
        print(f"ShotDetector initialized. Thresholds: max_dist={self.max_player_ball_dist}m, min_accel={self.min_ball_accel}m/s²")

    def reset_clip(self):
        self.prev_ball_pos = None
        self.prev_frame_time = None
        self.ball_trail = []
        self.last_shot_time = -999
        print("ShotDetector reset for new clip.")

    def compute_ball_acceleration(self, current_pos: np.ndarray, frame_time: float) -> tuple:
        """
        Compute ball acceleration vector from position history.
        Args:
            current_pos: (2,) real-world (x,y) meters
            frame_time: timestamp in seconds
        Returns:
            (accel_mag, accel_vector) or (0.0, np.zeros(2)) if not enough data
        """
        if self.prev_ball_pos is None or self.prev_frame_time is None:
            self.prev_ball_pos = current_pos
            self.prev_frame_time = frame_time
            self.ball_trail.append(current_pos)
            return 0.0, np.zeros(2)
        
        # Time delta
        dt = frame_time - self.prev_frame_time
        if dt <= 0:
            return 0.0, np.zeros(2)
        
        # Velocity: (current - prev) / dt
        velocity = (current_pos - self.prev_ball_pos) / dt
        
        # Update trail for acceleration smoothing
        self.ball_trail.append(current_pos)
        if len(self.ball_trail) > self.trail_length:
            self.ball_trail.pop(0)
        
        # Acceleration: if we have at least 2 velocity samples
        if len(self.ball_trail) >= 2:
            # Simple finite difference for acceleration
            prev_pos_trail = self.ball_trail[-2]
            accel_vector = (current_pos - 2*self.prev_ball_pos + prev_pos_trail) / (dt**2 + 1e-6)
            accel_mag = np.linalg.norm(accel_vector)
        else:
            accel_mag = 0.0
            accel_vector = np.zeros(2)
        
        # Update state
        self.prev_ball_pos = current_pos
        self.prev_frame_time = frame_time
        
        return accel_mag, accel_vector

    def is_shot(self, 
                player_pos: np.ndarray,  # Real-world (x,y) meters (player foot)
                ball_pos: np.ndarray,     # Real-world (x,y) meters (ball center)
                ball_accel_mag: float,    # m/s²
                goal_pos: np.ndarray,      # Real-world (x,y) meters (goal center)
                player_team_id: int) -> tuple:
        """
        Core shot detection heuristic.
        Returns:
            (is_shot: bool, confidence: float, reason: str)
        """
        # 1. Check player-ball proximity
        player_ball_dist = np.linalg.norm(player_pos - ball_pos)
        if player_ball_dist > self.max_player_ball_dist:
            return False, 0.0, f"Player-ball distance {player_ball_dist:.2f}m > threshold"
        
        # 2. Check ball acceleration (sudden speed increase = shot)
        if ball_accel_mag < self.min_ball_accel:
            return False, 0.0, f"Ball acceleration {ball_accel_mag:.2f}m/s² < threshold"
        
        # 3. Check direction toward goal (dot product of ball movement and goal vector)
        ball_to_goal = goal_pos - ball_pos
        ball_to_goal_norm = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-6)
        
        # Approximate ball movement direction from acceleration vector
        # (In real implementation, use velocity vector if available)
        # For now, use player-to-ball vector as proxy for shot direction
        player_to_ball = ball_pos - player_pos
        player_to_ball_norm = player_to_ball / (np.linalg.norm(player_to_ball) + 1e-6)
        
        # Direction alignment: dot product >0.5 means roughly toward goal
        direction_alignment = np.dot(player_to_ball_norm, ball_to_goal_norm)
        if direction_alignment < 0.3:  # Ball moving away from goal
            return False, 0.0, f"Direction alignment {direction_alignment:.2f} < 0.3 (away from goal)"
        
        # 4. Check ball is reasonably close to goal (not a pass from midfield)
        ball_to_goal_dist = np.linalg.norm(ball_to_goal)
        if ball_to_goal_dist > self.goal_proximity:
            return False, 0.0, f"Ball-goal distance {ball_to_goal_dist:.2f}m > {self.goal_proximity}m"
        
        # Shot detected! Calculate confidence based on how well thresholds are met
        dist_score = 1.0 - (player_ball_dist / self.max_player_ball_dist)
        accel_score = min(ball_accel_mag / (self.min_ball_accel * 2), 1.0)
        direction_score = (direction_alignment + 1) / 2  # Normalize -1..1 to 0..1
        
        confidence = (dist_score * 0.4 + accel_score * 0.4 + direction_score * 0.2)
        return True, confidence, f"SHOT: dist={player_ball_dist:.2f}m, accel={ball_accel_mag:.2f}m/s², align={direction_alignment:.2f}"


    def detect_shot(self, ball_pos: np.ndarray, frame_time: float,
                accel_mag: float, closest_player: dict) -> tuple:
        """
        Wrapper chamado pelo pipeline. Agora usa is_shot() completo.
        closest_player deve conter 'foot_pos_real' — posição do pé em metros.
        """
        
        if 7.0 <= frame_time <= 12.0 or 32.0 <= frame_time <= 37.0 or 92.0 <= frame_time <= 96.0:
            team_id = closest_player.get('team', -1)
            goal_pos = np.array([0.0, 35.0]) if team_id == 1 else np.array([120.0, 35.0])
            player_pos = closest_player.get('foot_pos_real')
            goal_dist = np.linalg.norm(goal_pos - ball_pos)
            player_dist = np.linalg.norm(player_pos - ball_pos) if player_pos is not None else -1
            print(f"[DEBUG] t={frame_time:.2f}s | ball_pos=({ball_pos[0]:.1f}, {ball_pos[1]:.1f}) | team={team_id} | goal_pos={goal_pos} | goal_dist={goal_dist:.2f}m | player_dist={player_dist:.2f}m")
        
        if frame_time - self.last_shot_time < self.shot_cooldown:
            return False, {'confidence': 0.0, 'reason': 'cooldown'}

        team_id = closest_player.get('team', -1)
        goal_pos = np.array([0.0, 35.0]) if team_id == 1 else np.array([120.0, 35.0])

        # Posição do jogador em metros reais
        player_pos = closest_player.get('foot_pos_real')
        if player_pos is None:
            return False, {'confidence': 0.0, 'reason': 'no_player_pos'}

        is_shot_result, confidence, reason = self.is_shot(
            player_pos, ball_pos, accel_mag, goal_pos, team_id
        )

        if not is_shot_result:
            return False, {'confidence': confidence, 'reason': reason}

        self.last_shot_time = frame_time
        return True, {'confidence': confidence, 'reason': reason, 'goal_pos': goal_pos.tolist()}


if __name__ == "__main__":
    detector = ShotDetector()
    print("Module test passed. ShotDetector ready.")
    
    # Test with dummy data
    goal_pos = np.array([105.0, 34.0])  # Right goal center (FIFA pitch: 105x68m)
    player_pos = np.array([100.0, 34.0])  # 5m from goal
    ball_pos = np.array([101.0, 34.0])   # 4m from goal
    
    is_shot, conf, reason = detector.is_shot(player_pos, ball_pos, 5.0, goal_pos, 0)
    print(f"Test shot: {is_shot}, conf={conf:.2f}, reason={reason}")
