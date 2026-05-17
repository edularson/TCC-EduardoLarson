import numpy as np
import cv2
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
from sports.configs.soccer import SoccerPitchConfiguration

# ── Dimensões reais do campo no sistema pipeline ──────────────────────────────
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0

# ── Paleta profissional (BGR) ─────────────────────────────────────────────────
COLORS = {
    'team0':      (200, 100,  30),   # azul cobalto escuro
    'team1':      ( 30,  30, 180),   # vermelho escuro
    'referee':    ( 40, 200, 200),   # amarelo escuro
    'ball':       (255, 255, 255),   # branco
    'gk_border':  (220, 220, 220),   # cinza claro
    'text_bg':    ( 15,  15,  15),   # quase preto
    'white':      (255, 255, 255),
    'black':      (  0,   0,   0),
}

# Cores do radar (profissional, baixa saturação)
RADAR_TEAM0_HEX = '1a4b8c'   # azul escuro
RADAR_TEAM1_HEX = '8c1a1a'   # vermelho escuro
RADAR_BG_HEX    = '1a3320'   # verde muito escuro

# Duração do overlay de chute em segundos
SHOT_OVERLAY_DURATION_S = 5.0


# ── helpers internos ──────────────────────────────────────────────────────────

def _draw_corner_bracket(frame, x1, y1, x2, y2, color, thickness=2, length=12):
    """Cantos destacados estilo videogame."""
    pts = [
        ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
        ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
        ((x2, y2 - length), (x2, y2), (x2 - length, y2)),
        ((x1 + length, y2), (x1, y2), (x1, y2 - length)),
    ]
    for p0, p1, p2 in pts:
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_label(frame, text, x, y, color, font_scale=0.42, thickness=1):
    """Label com fundo semitransparente."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    x1b, y1b = x - pad, y - th - pad
    x2b, y2b = x + tw + pad, y + baseline + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1b, y1b), (x2b, y2b), COLORS['text_bg'], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _to_radar_coords(xy_m: np.ndarray, config: SoccerPitchConfiguration) -> np.ndarray:
    """
    Converte metros do sistema pipeline (105x68) → espaço da biblioteca sports.
    Inverte Y para corrigir orientação (Y=0 no pipeline é topo, Y=0 na lib é base).
    """
    xy = np.atleast_2d(xy_m).astype(float)
    out = np.column_stack([
        xy[:, 0] / PITCH_LENGTH_M * config.length,
        # inversão do eixo Y: corrige minimapa de ponta-cabeça
        (1.0 - xy[:, 1] / PITCH_WIDTH_M) * config.width,
    ])
    return out


# ── API pública ───────────────────────────────────────────────────────────────

def draw_detections_on_frame(frame: np.ndarray, detections: dict) -> np.ndarray:
    """Desenha detecções in-place com estilo videogame."""

    # ── bola ─────────────────────────────────────────────────────────────────
    for (x1, y1, x2, y2, conf) in detections.get('balls', []):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max((x2 - x1), (y2 - y1)) // 2 + 4
        cv2.circle(frame, (cx, cy), r, COLORS['ball'], 2, cv2.LINE_AA)
        cv2.line(frame, (cx - r - 4, cy), (cx - r + 2, cy), COLORS['ball'], 1, cv2.LINE_AA)
        cv2.line(frame, (cx + r - 2, cy), (cx + r + 4, cy), COLORS['ball'], 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r - 4), (cx, cy - r + 2), COLORS['ball'], 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + r - 2), (cx, cy + r + 4), COLORS['ball'], 1, cv2.LINE_AA)
        _draw_label(frame, f"Ball {conf:.2f}", x1, y1 - 6, COLORS['ball'])

    # ── jogadores ─────────────────────────────────────────────────────────────
    for det in detections.get('players', []):
        x1, y1, x2, y2, conf, cls_id, track_id, team_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS['team0'] if team_id == 0 else COLORS['team1']
        _draw_corner_bracket(frame, x1, y1, x2, y2, color, thickness=2, length=14)
        cx = (x1 + x2) // 2
        cv2.circle(frame, (cx, y2), 3, color, -1, cv2.LINE_AA)
        _draw_label(frame, f"#{track_id}", x1, y2 + 14, color)

    # ── goleiros ──────────────────────────────────────────────────────────────
    for det in detections.get('goalkeepers', []):
        x1, y1, x2, y2, conf, cls_id, track_id, team_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS['team0'] if team_id == 0 else COLORS['team1']
        _draw_corner_bracket(frame, x1, y1, x2, y2, COLORS['gk_border'], thickness=3, length=16)
        _draw_corner_bracket(frame, x1+2, y1+2, x2-2, y2-2, color, thickness=2, length=12)
        _draw_label(frame, f"GK#{track_id}", x1, y2 + 14, COLORS['gk_border'])

    # ── árbitros ──────────────────────────────────────────────────────────────
    for det in detections.get('referees', []):
        x1, y1, x2, y2, conf, cls_id, track_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        _draw_corner_bracket(frame, x1, y1, x2, y2, COLORS['referee'], thickness=1, length=10)
        _draw_label(frame, "REF", x1, y2 + 14, COLORS['referee'])

    return frame


def draw_shot_event(frame: np.ndarray, shot_event: tuple,
                    xg_value: float = None,
                    elapsed_s: float = 0.0) -> np.ndarray:
    """
    Overlay de chute no canto superior direito.
    elapsed_s: segundos desde a detecção — usado para fade out nos últimos 1s.
    """
    track_id, conf, reason = shot_event
    h, w = frame.shape[:2]
    xg = xg_value if xg_value is not None else 0.0

    # fade out nos últimos 1s
    alpha_base = 0.72
    if elapsed_s > SHOT_OVERLAY_DURATION_S - 1.0:
        fade = 1.0 - (elapsed_s - (SHOT_OVERLAY_DURATION_S - 1.0))
        alpha_base = max(0.1, alpha_base * fade)

    # cor por intensidade de xG
    if xg >= 0.3:
        accent = (0, 50, 255)
    elif xg >= 0.1:
        accent = (0, 140, 255)
    else:
        accent = (80, 200, 80)

    pw, ph = 300, 96
    px = w - pw - 16
    py = 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha_base, frame, 1.0 - alpha_base, 0, frame)

    cv2.rectangle(frame, (px, py), (px + 5, py + ph), accent, -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), accent, 1, cv2.LINE_AA)

    cv2.putText(frame, "SHOT DETECTED", (px + 14, py + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLORS['white'], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Player #{track_id}  |  conf {conf:.2f}",
                (px + 14, py + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

    # barra de xG
    bar_x, bar_y = px + 14, py + 62
    bar_w, bar_h = pw - 80, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
    filled = int(bar_w * min(xg / 0.5, 1.0))
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), accent, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
    cv2.putText(frame, f"xG {xg:.3f}", (bar_x + bar_w + 8, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, accent, 1, cv2.LINE_AA)

    # timer de duração
    remaining = max(0.0, SHOT_OVERLAY_DURATION_S - elapsed_s)
    cv2.putText(frame, f"{remaining:.1f}s", (px + pw - 38, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1, cv2.LINE_AA)

    return frame


def create_radar_view(pitch_players_xy_m: np.ndarray,
                      players_team_ids: np.ndarray,
                      pitch_ball_xy_m: np.ndarray = None,
                      config: SoccerPitchConfiguration = None) -> np.ndarray:
    """
    Minimapa com cores profissionais e Y corrigido.
    Escala: metros pipeline (105x68) → espaço da biblioteca sports.
    """
    if config is None:
        config = SoccerPitchConfiguration()

    radar = draw_pitch(config, background_color=sv.Color.from_hex(RADAR_BG_HEX))

    team_0_mask = players_team_ids == 0
    team_1_mask = players_team_ids == 1

    if np.any(team_0_mask):
        xy_t0 = _to_radar_coords(pitch_players_xy_m[team_0_mask], config)
        radar  = draw_points_on_pitch(
            config=config, xy=xy_t0,
            face_color=sv.Color.from_hex(RADAR_TEAM0_HEX),
            edge_color=sv.Color.WHITE, radius=16, pitch=radar,
        )
    if np.any(team_1_mask):
        xy_t1 = _to_radar_coords(pitch_players_xy_m[team_1_mask], config)
        radar  = draw_points_on_pitch(
            config=config, xy=xy_t1,
            face_color=sv.Color.from_hex(RADAR_TEAM1_HEX),
            edge_color=sv.Color.WHITE, radius=16, pitch=radar,
        )

    if pitch_ball_xy_m is not None:
        ball_xy = _to_radar_coords(pitch_ball_xy_m.reshape(1, -1), config)
        radar   = draw_points_on_pitch(
            config=config, xy=ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK, radius=10, pitch=radar,
        )

    if np.any(team_0_mask) and np.any(team_1_mask):
        radar = draw_pitch_voronoi_diagram(
            config=config,
            team_1_xy=_to_radar_coords(pitch_players_xy_m[team_0_mask], config),
            team_2_xy=_to_radar_coords(pitch_players_xy_m[team_1_mask], config),
            team_1_color=sv.Color.from_hex(RADAR_TEAM0_HEX),
            team_2_color=sv.Color.from_hex(RADAR_TEAM1_HEX),
            pitch=radar,
        )

    return radar


def draw_radar_on_frame(frame: np.ndarray,
                        pitch_players_xy_m: np.ndarray,
                        players_team_ids: np.ndarray,
                        pitch_ball_xy_m: np.ndarray = None,
                        scale: float = 0.28,
                        team0_name: str = "Team A",
                        team1_name: str = "Team B") -> np.ndarray:
    """
    Desenha o minimapa centralizado na parte inferior do frame.
    Inclui legenda de times. Sempre visível independente da bola.
    """
    config = SoccerPitchConfiguration()
    radar  = create_radar_view(pitch_players_xy_m, players_team_ids,
                               pitch_ball_xy_m, config)

    rh, rw  = radar.shape[:2]
    new_w   = int(rw * scale)
    new_h   = int(rh * scale)
    radar_s = cv2.resize(radar, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h_fr, w_fr = frame.shape[:2]

    # centralizado, 10px acima da borda inferior
    x_off = (w_fr - new_w) // 2
    y_off = h_fr - new_h - 10

    # fundo com borda fina
    pad = 2
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x_off - pad, y_off - pad),
                  (x_off + new_w + pad, y_off + new_h + pad),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame,
                  (x_off - pad, y_off - pad),
                  (x_off + new_w + pad, y_off + new_h + pad),
                  (80, 80, 80), 1)

    frame[y_off:y_off + new_h, x_off:x_off + new_w] = radar_s

    # legenda
    font       = cv2.FONT_HERSHEY_SIMPLEX
    leg_y      = y_off + new_h + 16
    dot_r      = 5
    team0_bgr  = tuple(int(RADAR_TEAM0_HEX[i:i+2], 16) for i in (4, 2, 0))
    team1_bgr  = tuple(int(RADAR_TEAM1_HEX[i:i+2], 16) for i in (4, 2, 0))

    # time 0
    lx0 = x_off
    cv2.circle(frame, (lx0 + dot_r, leg_y - dot_r), dot_r, team0_bgr, -1, cv2.LINE_AA)
    cv2.putText(frame, team0_name, (lx0 + dot_r * 2 + 4, leg_y),
                font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # time 1
    lx1 = x_off + new_w // 2
    cv2.circle(frame, (lx1 + dot_r, leg_y - dot_r), dot_r, team1_bgr, -1, cv2.LINE_AA)
    cv2.putText(frame, team1_name, (lx1 + dot_r * 2 + 4, leg_y),
                font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def draw_pitch_with_xg(radar_img: np.ndarray,
                        shot_pos_m: np.ndarray,
                        xg_value: float,
                        config: SoccerPitchConfiguration = None) -> np.ndarray:
    """Marcador de chute no mapa de campo estático (shot_map.png)."""
    if config is None:
        config = SoccerPitchConfiguration()

    shot_xy   = _to_radar_coords(shot_pos_m.reshape(1, -1), config)
    annotated = draw_points_on_pitch(
        config=config, xy=shot_xy,
        face_color=sv.Color.RED,
        edge_color=sv.Color.BLACK, radius=20, pitch=radar_img,
    )

    text_x = int(shot_xy[0, 0] * annotated.shape[1] / config.length)
    text_y = int(shot_xy[0, 1] * annotated.shape[0] / config.width)
    cv2.putText(annotated, f"xG={xg_value:.3f}",
                (text_x + 15, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return annotated