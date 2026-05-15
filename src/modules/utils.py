import numpy as np
import cv2
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram
from sports.configs.soccer import SoccerPitchConfiguration

# ── Dimensões reais do campo no sistema pipeline ──────────────────────────────
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0

# ── Paleta de cores (BGR) ─────────────────────────────────────────────────────
COLORS = {
    'team0':      (0,   191, 255),   # azul claro
    'team1':      (255,  20, 147),   # rosa/magenta
    'referee':    (0,   215, 255),   # amarelo
    'ball':       (255, 255,   0),   # ciano
    'gk_border':  (255, 255, 255),   # branco
    'text_bg':    (20,   20,  20),   # quase preto
    'white':      (255, 255, 255),
    'black':      (0,     0,   0),
}


# ── helpers internos ──────────────────────────────────────────────────────────

def _draw_corner_bracket(frame, x1, y1, x2, y2, color, thickness=2, length=12):
    """Desenha apenas os cantos de um retângulo — estilo FIFA/videogame."""
    pts = [
        # canto superior esquerdo
        ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
        # canto superior direito
        ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
        # canto inferior direito
        ((x2, y2 - length), (x2, y2), (x2 - length, y2)),
        # canto inferior esquerdo
        ((x1 + length, y2), (x1, y2), (x1, y2 - length)),
    ]
    for p0, p1, p2 in pts:
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_label(frame, text, x, y, color, font_scale=0.42, thickness=1):
    """Label com fundo semitransparente arredondado."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    x1b, y1b = x - pad, y - th - pad
    x2b, y2b = x + tw + pad, y + baseline + pad

    # fundo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1b, y1b), (x2b, y2b), COLORS['text_bg'], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _to_radar_coords(xy_m: np.ndarray, config: SoccerPitchConfiguration) -> np.ndarray:
    """
    Converte metros do sistema pipeline (105x68) para o espaço da biblioteca
    sports (length x width em centímetros, ex: 12000x7000).

    Problema original: multiplicar por 100 assumia campo 120x70m, mas o
    sistema pipeline usa 105x68m — jogadores apareciam fora das linhas corretas.
    """
    xy = np.atleast_2d(xy_m).astype(float)
    out = np.column_stack([
        xy[:, 0] / PITCH_LENGTH_M * config.length,
        xy[:, 1] / PITCH_WIDTH_M  * config.width,
    ])
    return out


# ── API pública ───────────────────────────────────────────────────────────────

def draw_detections_on_frame(frame: np.ndarray, detections: dict) -> np.ndarray:
    """
    Desenha detecções in-place com estilo videogame:
    - Cantos destacados em vez de retângulo completo
    - Labels com fundo semitransparente
    - Bola com crosshair
    """
    # ── bola ─────────────────────────────────────────────────────────────────
    for (x1, y1, x2, y2, conf) in detections.get('balls', []):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max((x2 - x1), (y2 - y1)) // 2 + 4

        # círculo externo
        cv2.circle(frame, (cx, cy), r, COLORS['ball'], 2, cv2.LINE_AA)
        # crosshair
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

        # ponto de pé (centro-base)
        cx = (x1 + x2) // 2
        cv2.circle(frame, (cx, y2), 3, color, -1, cv2.LINE_AA)

        _draw_label(frame, f"#{track_id}", x1, y2 + 14, color)

    # ── goleiros ──────────────────────────────────────────────────────────────
    for det in detections.get('goalkeepers', []):
        x1, y1, x2, y2, conf, cls_id, track_id, team_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS['team0'] if team_id == 0 else COLORS['team1']

        # goleiro tem borda dupla para diferenciar
        _draw_corner_bracket(frame, x1, y1, x2, y2, COLORS['gk_border'], thickness=3, length=16)
        _draw_corner_bracket(frame, x1 + 2, y1 + 2, x2 - 2, y2 - 2, color, thickness=2, length=12)

        _draw_label(frame, f"GK#{track_id}", x1, y2 + 14, COLORS['gk_border'])

    # ── árbitros ──────────────────────────────────────────────────────────────
    for det in detections.get('referees', []):
        x1, y1, x2, y2, conf, cls_id, track_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS['referee']

        _draw_corner_bracket(frame, x1, y1, x2, y2, color, thickness=1, length=10)
        _draw_label(frame, f"REF", x1, y2 + 14, color)

    return frame


def draw_shot_event(frame: np.ndarray, shot_event: tuple, xg_value: float = None) -> np.ndarray:
    """
    Overlay de chute profissional no canto superior direito.

    Design:
    - Painel semitransparente com borda colorida por intensidade de xG
    - Barra de xG preenchida proporcional ao valor
    - Não ocupa o centro da tela
    """
    track_id, conf, reason = shot_event
    h, w = frame.shape[:2]

    xg = xg_value if xg_value is not None else 0.0

    # cor da borda por intensidade de xG
    if xg >= 0.3:
        accent = (0, 50, 255)    # vermelho — chute perigoso
    elif xg >= 0.1:
        accent = (0, 165, 255)   # laranja — moderado
    else:
        accent = (0, 220, 100)   # verde — baixo xG

    # dimensões do painel
    pw, ph = 280, 90
    px = w - pw - 16
    py = 16

    # fundo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # borda colorida esquerda (acento)
    cv2.rectangle(frame, (px, py), (px + 5, py + ph), accent, -1)

    # borda externa fina
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), accent, 1, cv2.LINE_AA)

    # texto "SHOT"
    cv2.putText(frame, "SHOT DETECTED", (px + 14, py + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLORS['white'], 1, cv2.LINE_AA)

    # player ID
    cv2.putText(frame, f"Player #{track_id}  |  conf {conf:.2f}",
                (px + 14, py + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (180, 180, 180), 1, cv2.LINE_AA)

    # barra de xG
    bar_x, bar_y = px + 14, py + 58
    bar_w, bar_h = pw - 28, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled = int(bar_w * min(xg / 0.5, 1.0))   # escala: 0.5 xG = barra cheia
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      accent, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (100, 100, 100), 1)

    # valor de xG
    xg_text = f"xG = {xg:.3f}"
    cv2.putText(frame, xg_text, (bar_x + bar_w + 6, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)

    return frame


def create_radar_view(pitch_players_xy_m: np.ndarray,
                      players_team_ids: np.ndarray,
                      pitch_ball_xy_m: np.ndarray = None,
                      config: SoccerPitchConfiguration = None) -> np.ndarray:
    """
    Minimapa com posições em metros do sistema pipeline (105x68).
    Corrige remapeamento para o espaço da biblioteca sports (120x70 equiv).
    """
    if config is None:
        config = SoccerPitchConfiguration()

    radar = draw_pitch(config, background_color=sv.Color.from_hex('2d6a2d'))

    team_0_mask = players_team_ids == 0
    team_1_mask = players_team_ids == 1

    if np.any(team_0_mask):
        xy_t0 = _to_radar_coords(pitch_players_xy_m[team_0_mask], config)
        radar  = draw_points_on_pitch(
            config=config, xy=xy_t0,
            face_color=sv.Color.from_hex('00BFFF'),
            edge_color=sv.Color.BLACK, radius=16, pitch=radar,
        )
    if np.any(team_1_mask):
        xy_t1 = _to_radar_coords(pitch_players_xy_m[team_1_mask], config)
        radar  = draw_points_on_pitch(
            config=config, xy=xy_t1,
            face_color=sv.Color.from_hex('FF1493'),
            edge_color=sv.Color.BLACK, radius=16, pitch=radar,
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
            team_1_color=sv.Color.from_hex('00BFFF'),
            team_2_color=sv.Color.from_hex('FF1493'),
            pitch=radar,
        )

    return radar


def draw_pitch_with_xg(radar_img: np.ndarray,
                        shot_pos_m: np.ndarray,
                        xg_value: float,
                        config: SoccerPitchConfiguration = None) -> np.ndarray:
    """Adiciona marcador de chute/xG no mapa de campo."""
    if config is None:
        config = SoccerPitchConfiguration()

    shot_xy = _to_radar_coords(shot_pos_m.reshape(1, -1), config)

    annotated = draw_points_on_pitch(
        config=config, xy=shot_xy,
        face_color=sv.Color.RED,
        edge_color=sv.Color.BLACK, radius=20, pitch=radar_img,
    )

    # posição do texto proporcional à imagem
    text_x = int(shot_xy[0, 0] * annotated.shape[1] / config.length)
    text_y = int(shot_xy[0, 1] * annotated.shape[0] / config.width)
    cv2.putText(annotated, f"xG={xg_value:.3f}",
                (text_x + 15, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return annotated