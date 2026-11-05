import numpy as np
import cv2
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
from sports.configs.soccer import SoccerPitchConfiguration

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0

# ── Paleta FIFA/EA Sports ─────────────────────────────────────────────────────
# Cores únicas por função — sem ambiguidade
COLORS = {
    'team0':      ( 40, 200, 255),   # BGR para amarelo-ouro
    'team1':      (255, 160,  60),   # BGR para azul claro
    'referee':    (200, 200, 200),
    'ball':       (255, 255, 255),
    'gk_border':  (255, 255, 255),
    'text_bg':    (  0,   0,   0),
    'white':      (255, 255, 255),
    'black':      (  0,   0,   0),
    'xg_high':    ( 30,  30, 220),   # vermelho — perigoso
    'xg_mid':     ( 30, 140, 255),   # laranja — moderado
    'xg_low':     ( 60, 200,  80),   # verde — baixo xG
}

# ── helpers internos ──────────────────────────────────────────────────────────

def _draw_corner_bracket(frame, x1, y1, x2, y2, color, thickness=2, length=12):
    pts = [
        ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
        ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
        ((x2, y2 - length), (x2, y2), (x2 - length, y2)),
        ((x1 + length, y2), (x1, y2), (x1, y2 - length)),
    ]
    for p0, p1, p2 in pts:
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_label(frame, text, x, y, color, font_scale=0.40, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad), COLORS['text_bg'], -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale,
                color, thickness, cv2.LINE_AA)


def _xg_color(xg: float):
    if xg >= 0.3:
        return COLORS['xg_high']
    elif xg >= 0.1:
        return COLORS['xg_mid']
    return COLORS['xg_low']


def _to_radar_coords(xy_m: np.ndarray,
                     config: SoccerPitchConfiguration) -> np.ndarray:
    xy = np.atleast_2d(xy_m).astype(float)
    return np.column_stack([
        xy[:, 0] / PITCH_LENGTH_M * config.length,
        (1.0 - xy[:, 1] / PITCH_WIDTH_M) * config.width,
    ])


# ── API pública ───────────────────────────────────────────────────────────────

def draw_detections_on_frame(frame: np.ndarray,
                              detections: dict) -> np.ndarray:
    # ── bola ─────────────────────────────────────────────────────────────────
    for (x1, y1, x2, y2, conf) in detections.get('balls', []):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max((x2 - x1), (y2 - y1)) // 2 + 4
        cv2.circle(frame, (cx, cy), r, COLORS['ball'], 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, COLORS['ball'], -1, cv2.LINE_AA)
        # crosshair minimalista
        cv2.line(frame, (cx - r - 5, cy), (cx - r + 2, cy),
                 COLORS['ball'], 1, cv2.LINE_AA)
        cv2.line(frame, (cx + r - 2, cy), (cx + r + 5, cy),
                 COLORS['ball'], 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r - 5), (cx, cy - r + 2),
                 COLORS['ball'], 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + r - 2), (cx, cy + r + 5),
                 COLORS['ball'], 1, cv2.LINE_AA)

    # ── jogadores ─────────────────────────────────────────────────────────────
    for det in detections.get('players', []):
        x1, y1, x2, y2, conf, cls_id, track_id, team_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS['team0'] if team_id == 0 else COLORS['team1']
        _draw_corner_bracket(frame, x1, y1, x2, y2, color, thickness=2, length=13)
        cx = (x1 + x2) // 2
        cv2.circle(frame, (cx, y2), 3, color, -1, cv2.LINE_AA)
        _draw_label(frame, f"#{track_id}", x1, y2 + 14, color)

    # ── goleiros ──────────────────────────────────────────────────────────────
    for det in detections.get('goalkeepers', []):
        x1, y1, x2, y2, conf, cls_id, track_id, team_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS['team0'] if team_id == 0 else COLORS['team1']
        _draw_corner_bracket(frame, x1, y1, x2, y2,
                             COLORS['gk_border'], thickness=3, length=16)
        _draw_corner_bracket(frame, x1+2, y1+2, x2-2, y2-2,
                             color, thickness=2, length=12)
        _draw_label(frame, f"GK#{track_id}", x1, y2 + 14, COLORS['gk_border'])

    # ── árbitros ──────────────────────────────────────────────────────────────
    for det in detections.get('referees', []):
        x1, y1, x2, y2, conf, cls_id, track_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        _draw_corner_bracket(frame, x1, y1, x2, y2,
                             COLORS['referee'], thickness=1, length=10)

    return frame


def draw_shot_overlay(frame: np.ndarray,
                      track_id: int,
                      xg_value: float,
                      alpha: float = 1.0) -> np.ndarray:
    """
    Overlay de chute estilo FIFA — canto superior direito.
    alpha: 1.0 = totalmente visível, 0.0 = invisível (fade out).
    """
    if alpha <= 0.01:
        return frame

    h, w = frame.shape[:2]
    xg   = xg_value if xg_value is not None else 0.0
    accent = _xg_color(xg)

    # escurece accent proporcionalmente ao fade
    def fade_color(c):
        return tuple(int(v * alpha) for v in c)

    pw, ph = 260, 80
    px = w - pw - 20
    py = 20

    # painel de fundo
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
    panel_alpha = 0.75 * alpha
    cv2.addWeighted(overlay, panel_alpha, frame, 1 - panel_alpha, 0, frame)

    # borda esquerda colorida
    bar_color = fade_color(accent)
    cv2.rectangle(frame, (px, py), (px + 4, py + ph), bar_color, -1)

    # linha superior fina
    cv2.line(frame, (px + 4, py), (px + pw, py), bar_color, 1, cv2.LINE_AA)

    # "SHOT" em caps — fonte limpa
    font = cv2.FONT_HERSHEY_SIMPLEX
    white_faded = fade_color(COLORS['white'])
    gray_faded  = fade_color((160, 160, 160))

    cv2.putText(frame, "SHOT", (px + 14, py + 22),
                font, 0.65, white_faded, 1, cv2.LINE_AA)

    # xG em destaque
    xg_str = f"xG  {xg:.3f}"
    cv2.putText(frame, xg_str, (px + 14, py + 46),
                font, 0.50, bar_color, 1, cv2.LINE_AA)

    # player ID menor
    cv2.putText(frame, f"Player #{track_id}", (px + 14, py + 64),
                font, 0.36, gray_faded, 1, cv2.LINE_AA)

    # barra de xG — fina, elegante
    bx, by = px + 14, py + 70
    bw, bh = pw - 28, 5
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    filled = int(bw * min(xg / 0.5, 1.0))
    if filled > 0:
        cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), bar_color, -1)

    return frame


def create_radar_view(pitch_players_xy_m: np.ndarray,
                      players_team_ids: np.ndarray,
                      pitch_ball_xy_m: np.ndarray = None,
                      config: SoccerPitchConfiguration = None) -> np.ndarray:
    if config is None:
        config = SoccerPitchConfiguration()

    radar = draw_pitch(config, background_color=sv.Color.from_hex('1a3d1a'))

    team_0_mask = players_team_ids == 0
    team_1_mask = players_team_ids == 1

    if np.any(team_0_mask):
        radar = draw_points_on_pitch(
            config=config,
            xy=_to_radar_coords(pitch_players_xy_m[team_0_mask], config),
            face_color=sv.Color.from_hex('FFC828'),
            edge_color=sv.Color.BLACK, radius=14, pitch=radar,
        )
    if np.any(team_1_mask):
        radar = draw_points_on_pitch(
            config=config,
            xy=_to_radar_coords(pitch_players_xy_m[team_1_mask], config),
            face_color=sv.Color.from_hex('3CA0FF'),
            edge_color=sv.Color.BLACK, radius=14, pitch=radar,
        )
    if pitch_ball_xy_m is not None:
        radar = draw_points_on_pitch(
            config=config,
            xy=_to_radar_coords(pitch_ball_xy_m.reshape(1, -1), config),
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK, radius=9, pitch=radar,
        )
    if np.any(team_0_mask) and np.any(team_1_mask):
        radar = draw_pitch_voronoi_diagram(
            config=config,
            team_1_xy=_to_radar_coords(pitch_players_xy_m[team_0_mask], config),
            team_2_xy=_to_radar_coords(pitch_players_xy_m[team_1_mask], config),
            team_1_color=sv.Color.from_hex('FFC828'),
            team_2_color=sv.Color.from_hex('3CA0FF'),
            pitch=radar,
        )

    return radar


def draw_pitch_with_xg(radar_img: np.ndarray,
                        shot_pos_m: np.ndarray,
                        xg_value: float,
                        config: SoccerPitchConfiguration = None) -> np.ndarray:
    if config is None:
        config = SoccerPitchConfiguration()

    shot_xy   = _to_radar_coords(shot_pos_m.reshape(1, -1), config)
    annotated = draw_points_on_pitch(
        config=config, xy=shot_xy,
        face_color=sv.Color.RED,
        edge_color=sv.Color.BLACK, radius=18, pitch=radar_img,
    )

    text_x = int(shot_xy[0, 0] * annotated.shape[1] / config.length)
    text_y = int(shot_xy[0, 1] * annotated.shape[0] / config.width)
    cv2.putText(annotated, f"xG={xg_value:.3f}",
                (text_x + 15, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)

    return annotated

def generate_heatmap(positions_xy_m: np.ndarray,
                     grid_w: int = 105,
                     grid_h: int = 68,
                     blur_kernel: int = 21) -> np.ndarray:
    """
    Gera mapa de densidade 2D a partir de posições em metros (sistema pipeline).
    Retorna array float32 normalizado [0, 1] no espaço da grade.
    """
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

    for x, y in positions_xy_m:
        px = int(np.clip(x / PITCH_LENGTH_M * grid_w, 0, grid_w - 1))
        py = int(np.clip((1.0 - y / PITCH_WIDTH_M) * grid_h, 0, grid_h - 1))
        heatmap[py, px] += 1.0

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def draw_heatmap_on_pitch(positions_xy_m: np.ndarray,
                           color_bgr: tuple,
                           config: SoccerPitchConfiguration = None,
                           blur_kernel: int = 21,
                           alpha: float = 0.55) -> np.ndarray:
    """
    Renderiza heatmap de posições sobre o campo 2D da biblioteca sports.
    color_bgr: cor base do heatmap (ex: COLORS['team0']).
    """
    if config is None:
        config = SoccerPitchConfiguration()

    base = draw_pitch(config, background_color=sv.Color.from_hex('1a3d1a'))
    h, w = base.shape[:2]

    if len(positions_xy_m) == 0:
        return base

    hm = generate_heatmap(positions_xy_m, grid_w=w, grid_h=h,
                           blur_kernel=blur_kernel)

    # Converte para colormap: intensidade → cor do time
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]  # BGR → RGB
    colored[:, :, 0] = (hm * b).astype(np.uint8)
    colored[:, :, 1] = (hm * g).astype(np.uint8)
    colored[:, :, 2] = (hm * r).astype(np.uint8)

    mask = (hm > 0.05).astype(np.float32)
    result = base.copy()
    for c in range(3):
        result[:, :, c] = np.clip(
            base[:, :, c] * (1 - mask * alpha) + colored[:, :, c] * mask * alpha,
            0, 255
        ).astype(np.uint8)

    return result


def draw_possession_bar(frame: np.ndarray,
                         possession: dict,
                         color_team0: tuple = None,
                         color_team1: tuple = None) -> np.ndarray:
    """
    Barra de posse de bola no topo do frame — estilo broadcast.
    possession: {'team0': float, 'team1': float} — valores em [0, 1] somando 1.
    """
    if color_team0 is None:
        color_team0 = COLORS['team0']
    if color_team1 is None:
        color_team1 = COLORS['team1']

    h, w = frame.shape[:2]
    bar_h  = 28
    bar_w  = 320
    bx     = (w - bar_w) // 2
    by     = 12

    t0 = possession.get('team0', 0.5)
    t1 = possession.get('team1', 0.5)
    total = t0 + t1
    if total < 1e-6:
        t0, t1 = 0.5, 0.5
        total = 1.0
    t0_norm = t0 / total
    t1_norm = t1 / total

    split = int(bar_w * t0_norm)

    # fundo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx - 60, by - 4), (bx + bar_w + 60, by + bar_h + 4),
                  (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # barra time 0
    cv2.rectangle(frame, (bx, by), (bx + split, by + bar_h), color_team0, -1)
    # barra time 1
    cv2.rectangle(frame, (bx + split, by), (bx + bar_w, by + bar_h), color_team1, -1)
    # borda
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # percentual time 0 — à esquerda da barra
    t0_str = f"{t0_norm * 100:.0f}%"
    cv2.putText(frame, t0_str, (bx - 55, by + 19),
                font, 0.55, color_team0, 1, cv2.LINE_AA)

    # percentual time 1 — à direita da barra
    t1_str = f"{t1_norm * 100:.0f}%"
    cv2.putText(frame, t1_str, (bx + bar_w + 8, by + 19),
                font, 0.55, color_team1, 1, cv2.LINE_AA)

    # label central
    cv2.putText(frame, "POSSE", (bx + bar_w // 2 - 20, by + 19),
                font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
