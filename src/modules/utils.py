import numpy as np
import cv2
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
from sports.configs.soccer import SoccerPitchConfiguration 

def draw_detections_on_frame(frame: np.ndarray, detections: dict) -> np.ndarray:
    annotated = frame.copy()
    colors = {
        'ball': (0, 255, 255),
        'player_team0': (255, 191, 0),
        'player_team1': (147, 20, 255),
        'goalkeeper_team0': (255, 191, 0),
        'goalkeeper_team1': (147, 20, 255),
        'referee': (0, 205, 255)
    }
    for (x1, y1, x2, y2, conf) in detections.get('balls', []):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colors['ball'], 2)
        cv2.putText(annotated, f"Ball {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['ball'], 1)
    for (x1, y1, x2, y2, conf, cls_id, track_id, team_id) in detections.get('players', []):
        color = colors['player_team0'] if team_id == 0 else colors['player_team1']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"#{track_id}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for (x1, y1, x2, y2, conf, cls_id, track_id, team_id) in detections.get('goalkeepers', []):
        color = colors['goalkeeper_team0'] if team_id == 0 else colors['goalkeeper_team1']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"GK#{track_id}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for (x1, y1, x2, y2, conf, cls_id, track_id) in detections.get('referees', []):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colors['referee'], 2)
        cv2.putText(annotated, f"REF#{track_id}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['referee'], 1)
    return annotated

def draw_shot_event(frame: np.ndarray, shot_event: tuple, xg_value: float = None) -> np.ndarray:
    annotated = frame.copy()
    track_id, conf, reason = shot_event
    h, w = frame.shape[:2]
    cv2.circle(annotated, (w//2, h//2), 50, (0, 0, 255), -1)
    cv2.putText(annotated, "SHOT!", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    if xg_value is not None:
        cv2.putText(annotated, f"xG={xg_value:.3f}", (w//2 - 100, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return annotated

def create_radar_view(pitch_players_xy: np.ndarray, players_team_ids: np.ndarray, 
                     pitch_ball_xy: np.ndarray = None, config: SoccerPitchConfiguration = None) -> np.ndarray:
    if config is None:
        config = SoccerPitchConfiguration()
    radar = draw_pitch(config, background_color=sv.Color.WHITE)
    team_0_mask = players_team_ids == 0
    team_1_mask = players_team_ids == 1
    if np.any(team_0_mask):
        radar = draw_points_on_pitch(config=config, xy=pitch_players_xy[team_0_mask], face_color=sv.Color.from_hex('00BFFF'), edge_color=sv.Color.BLACK, radius=16, pitch=radar)
    if np.any(team_1_mask):
        radar = draw_points_on_pitch(config=config, xy=pitch_players_xy[team_1_mask], face_color=sv.Color.from_hex('FF1493'), edge_color=sv.Color.BLACK, radius=16, pitch=radar)
    if pitch_ball_xy is not None:
        radar = draw_points_on_pitch(config=config, xy=pitch_ball_xy.reshape(1, -1), face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK, radius=10, pitch=radar)
    if np.any(team_0_mask) and np.any(team_1_mask):
        radar = draw_pitch_voronoi_diagram(config=config, team_1_xy=pitch_players_xy[team_0_mask], team_2_xy=pitch_players_xy[team_1_mask], team_1_color=sv.Color.from_hex('00BFFF'), team_2_color=sv.Color.from_hex('FF1493'), pitch=radar)
    return radar

def draw_pitch_with_xg(radar_img: np.ndarray, shot_pos: np.ndarray, xg_value: float, 
                        config: SoccerPitchConfiguration = None) -> np.ndarray:
    if config is None:
        config = SoccerPitchConfiguration()
    annotated = radar_img.copy()
    annotated = draw_points_on_pitch(config=config, xy=shot_pos.reshape(1, -1), face_color=sv.Color.RED, edge_color=sv.Color.BLACK, radius=20, pitch=annotated)
    text_x = int(shot_pos[0] * (annotated.shape[1] / config.width))
    text_y = int(shot_pos[1] * (annotated.shape[0] / config.length))
    cv2.putText(annotated, f"xG={xg_value:.3f}", (text_x + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return annotated
