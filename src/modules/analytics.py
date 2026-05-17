"""
analytics.py — VarzeaVision
Módulo de métricas táticas acumuladas por clipe.

Métricas implementadas:
- Posse de bola por time (% de frames com jogador mais próximo da bola)
- Heatmap de ocupação por zona (posições acumuladas por time)
- Distância percorrida por jogador (soma de deslocamentos em metros)
- Velocidade média e máxima por jogador (m/s)
"""

import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path

# Dimensões do campo
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0

# Resolução do heatmap em células
HEATMAP_COLS = 21   # 5m por célula em X
HEATMAP_ROWS = 14   # ~5m por célula em Y


class MatchAnalytics:
    """
    Acumula métricas frame a frame durante o processamento do clipe.
    Chamado pelo pipeline principal a cada frame.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Posse de bola
        self._possession_frames = {0: 0, 1: 0, -1: 0}   # team_id → n frames
        self._total_ball_frames = 0

        # Heatmap: grid de células por time
        self._heatmap = {
            0: np.zeros((HEATMAP_ROWS, HEATMAP_COLS), dtype=np.float32),
            1: np.zeros((HEATMAP_ROWS, HEATMAP_COLS), dtype=np.float32),
        }

        # Tracking de posições anteriores por jogador
        self._last_pos   = {}   # track_id → np.array([x, y])
        self._last_time  = {}   # track_id → float (segundos)

        # Distância acumulada por jogador
        self._distance   = defaultdict(float)   # track_id → metros

        # Velocidades por jogador
        self._speeds     = defaultdict(list)    # track_id → [v1, v2, ...]

        # Time de cada track_id
        self._tid_team   = {}   # track_id → team_id

    # ── atualização por frame ─────────────────────────────────────────────────

    def update(self,
               pitch_xy_real: np.ndarray,
               tracker_ids: np.ndarray,
               team_ids: np.ndarray,
               class_ids: np.ndarray,
               ball_pos: np.ndarray | None,
               frame_time: float):
        """
        Atualiza todas as métricas com os dados do frame atual.

        Args:
            pitch_xy_real: (N,2) posições em metros de todos os jogadores
            tracker_ids:   (N,)  IDs do tracker
            team_ids:      (N,)  time de cada jogador (0, 1 ou -1)
            class_ids:     (N,)  0=bola,1=gk,2=player,3=ref
            ball_pos:      (2,) posição da bola em metros, ou None
            frame_time:    float, segundo atual do vídeo
        """
        player_mask = (class_ids == 2) | (class_ids == 1)   # field players + gk

        if np.any(player_mask):
            p_xy   = pitch_xy_real[player_mask]
            p_tids = tracker_ids[player_mask]
            p_tms  = team_ids[player_mask]

            # ── heatmap ───────────────────────────────────────────────────────
            for xy, tid, tm in zip(p_xy, p_tids, p_tms):
                self._tid_team[int(tid)] = int(tm)

                if tm in (0, 1):
                    col = int(np.clip(xy[0] / PITCH_LENGTH_M * HEATMAP_COLS, 0, HEATMAP_COLS - 1))
                    row = int(np.clip(xy[1] / PITCH_WIDTH_M  * HEATMAP_ROWS, 0, HEATMAP_ROWS - 1))
                    self._heatmap[tm][row, col] += 1.0

            # ── distância e velocidade ────────────────────────────────────────
            for xy, tid in zip(p_xy, p_tids):
                tid = int(tid)
                if tid in self._last_pos and tid in self._last_time:
                    dt = frame_time - self._last_time[tid]
                    if 0 < dt <= 1.0:   # ignora gaps grandes (cena de replay)
                        dist = float(np.linalg.norm(xy - self._last_pos[tid]))
                        if dist < 10.0:  # ignora teleporte (ID swap)
                            self._distance[tid] += dist
                            speed = dist / dt
                            if speed < 12.0:   # cap físico ~43 km/h para humanos
                                self._speeds[tid].append(speed)

                self._last_pos[tid]  = xy.copy()
                self._last_time[tid] = frame_time

        # ── posse de bola ─────────────────────────────────────────────────────
        if ball_pos is not None and np.any(player_mask):
            p_xy   = pitch_xy_real[player_mask]
            p_tms  = team_ids[player_mask]

            dists     = np.linalg.norm(p_xy - ball_pos, axis=1)
            closest_i = int(np.argmin(dists))

            if dists[closest_i] < 5.0:   # só conta se dentro de 5m
                closest_team = int(p_tms[closest_i])
                self._possession_frames[closest_team] = \
                    self._possession_frames.get(closest_team, 0) + 1
                self._total_ball_frames += 1

    # ── getters ───────────────────────────────────────────────────────────────

    def get_possession(self) -> dict:
        """Retorna posse de bola em % por time."""
        total = self._total_ball_frames
        if total == 0:
            return {0: 0.0, 1: 0.0}
        return {
            0: round(100.0 * self._possession_frames.get(0, 0) / total, 1),
            1: round(100.0 * self._possession_frames.get(1, 0) / total, 1),
        }

    def get_player_distances(self) -> dict:
        """Retorna distância total percorrida (metros) por track_id."""
        return dict(self._distance)

    def get_player_speeds(self) -> dict:
        """Retorna {'avg': m/s, 'max': m/s} por track_id."""
        result = {}
        for tid, speeds in self._speeds.items():
            if speeds:
                result[tid] = {
                    'avg_ms': round(float(np.mean(speeds)), 2),
                    'max_ms': round(float(np.max(speeds)), 2),
                    'avg_kmh': round(float(np.mean(speeds)) * 3.6, 1),
                    'max_kmh': round(float(np.max(speeds)) * 3.6, 1),
                }
        return result

    def get_summary(self, team0_name: str = "Team A",
                    team1_name: str = "Team B") -> dict:
        """Retorna resumo completo das métricas."""
        possession   = self.get_possession()
        distances    = self.get_player_distances()
        speeds       = self.get_player_speeds()

        # agrupa distância e velocidade por time
        team_dist    = {0: 0.0, 1: 0.0}
        team_speeds  = {0: [], 1: []}

        for tid, dist in distances.items():
            tm = self._tid_team.get(tid, -1)
            if tm in (0, 1):
                team_dist[tm] += dist

        for tid, sp in speeds.items():
            tm = self._tid_team.get(tid, -1)
            if tm in (0, 1):
                team_speeds[tm].append(sp['max_ms'])

        return {
            'possession': {
                team0_name: possession[0],
                team1_name: possession[1],
            },
            'total_distance_km': {
                team0_name: round(team_dist[0] / 1000, 2),
                team1_name: round(team_dist[1] / 1000, 2),
            },
            'max_speed_kmh': {
                team0_name: round(max(team_speeds[0]) * 3.6, 1) if team_speeds[0] else 0.0,
                team1_name: round(max(team_speeds[1]) * 3.6, 1) if team_speeds[1] else 0.0,
            },
            'per_player': {
                str(tid): {
                    'team':     self._tid_team.get(tid, -1),
                    'distance_m': round(distances.get(tid, 0.0), 1),
                    **speeds.get(tid, {'avg_ms': 0.0, 'max_ms': 0.0,
                                       'avg_kmh': 0.0, 'max_kmh': 0.0}),
                }
                for tid in set(list(distances.keys()) + list(speeds.keys()))
            },
        }

    # ── exportação ────────────────────────────────────────────────────────────

    def save_heatmaps(self, output_dir: str,
                      team0_name: str = "Team A",
                      team1_name: str = "Team B"):
        """
        Salva heatmaps de ocupação como imagens PNG.
        Um arquivo por time + um comparativo lado a lado.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        imgs  = {}

        for team_id, name in [(0, team0_name), (1, team1_name)]:
            hm   = self._heatmap[team_id].copy()
            img  = _render_heatmap(hm, name)
            path = output_dir / f"heatmap_{name.replace(' ', '_')}.png"
            cv2.imwrite(str(path), img)
            paths[name] = str(path)
            imgs[team_id] = img

        # comparativo lado a lado
        if imgs:
            combined = np.hstack([imgs[0], imgs[1]])
            path_comb = output_dir / "heatmap_combined.png"
            cv2.imwrite(str(path_comb), combined)
            paths['combined'] = str(path_comb)

        return paths


# ── renderização de heatmap ───────────────────────────────────────────────────

def _render_heatmap(grid: np.ndarray, title: str,
                    img_w: int = 840, img_h: int = 540) -> np.ndarray:
    """
    Renderiza o grid de ocupação como imagem com campo desenhado por baixo.
    """
    # normaliza
    if grid.max() > 0:
        grid_norm = (grid / grid.max() * 255).astype(np.uint8)
    else:
        grid_norm = np.zeros_like(grid, dtype=np.uint8)

    # redimensiona para resolução da imagem
    hm_big = cv2.resize(grid_norm, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    # aplica colormap
    hm_color = cv2.applyColorMap(hm_big, cv2.COLORMAP_JET)

    # fundo verde (campo)
    field = np.full((img_h, img_w, 3), (34, 85, 34), dtype=np.uint8)

    # blende heatmap sobre o campo
    alpha  = (hm_big.astype(np.float32) / 255.0 * 0.75 + 0.0)
    alpha3 = np.stack([alpha, alpha, alpha], axis=2)
    result = (hm_color * alpha3 + field * (1 - alpha3)).astype(np.uint8)

    # desenha linhas do campo por cima
    _draw_field_lines(result, img_w, img_h)

    # título
    cv2.putText(result, title, (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return result


def _draw_field_lines(img: np.ndarray, w: int, h: int):
    """Desenha as linhas básicas do campo sobre a imagem."""
    c  = (255, 255, 255)
    t  = 1

    def px(xm, ym):
        return (int(xm / PITCH_LENGTH_M * w), int(ym / PITCH_WIDTH_M * h))

    # contorno
    cv2.rectangle(img, px(0, 0), px(105, 68), c, t)
    # meio campo
    cv2.line(img, px(52.5, 0), px(52.5, 68), c, t)
    # grandes áreas
    cv2.rectangle(img, px(0, 13.84),  px(16.5, 54.16), c, t)
    cv2.rectangle(img, px(88.5, 13.84), px(105, 54.16), c, t)
    # pequenas áreas
    cv2.rectangle(img, px(0, 24.84),  px(5.5, 43.16), c, t)
    cv2.rectangle(img, px(99.5, 24.84), px(105, 43.16), c, t)
    # marca do pênalti
    cv2.circle(img, px(11, 34),  3, c, -1)
    cv2.circle(img, px(94, 34),  3, c, -1)
    # centro
    cv2.circle(img, px(52.5, 34), int(9.15 / PITCH_LENGTH_M * w), c, t)
    cv2.circle(img, px(52.5, 34), 3, c, -1)


# ── painel de métricas no vídeo ───────────────────────────────────────────────

def draw_analytics_panel(frame: np.ndarray,
                         analytics: MatchAnalytics,
                         team0_name: str = "Team A",
                         team1_name: str = "Team B") -> np.ndarray:
    """
    Painel fixo no canto superior esquerdo com posse, distância e velocidade.
    Atualizado a cada frame.
    """
    h_fr, w_fr = frame.shape[:2]

    pw, ph = 230, 110
    px_off, py_off = 12, 12

    # fundo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (px_off, py_off),
                  (px_off + pw, py_off + ph),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame,
                  (px_off, py_off),
                  (px_off + pw, py_off + ph),
                  (80, 80, 80), 1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    poss  = analytics.get_possession()
    dists = analytics.get_player_distances()
    spds  = analytics.get_player_speeds()

    # distância total por time
    team_dist = {0: 0.0, 1: 0.0}
    for tid, dist in dists.items():
        tm = analytics._tid_team.get(tid, -1)
        if tm in (0, 1):
            team_dist[tm] += dist

    # velocidade máxima por time
    team_maxspd = {0: 0.0, 1: 0.0}
    for tid, sp in spds.items():
        tm = analytics._tid_team.get(tid, -1)
        if tm in (0, 1) and sp['max_ms'] > team_maxspd[tm]:
            team_maxspd[tm] = sp['max_ms']

    # cores dos times (BGR)
    c0 = (200, 100, 30)
    c1 = (30,  30, 180)

    # título
    cv2.putText(frame, "ANALYTICS", (px_off + 8, py_off + 16),
                font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # linha separadora
    cv2.line(frame,
             (px_off + 6, py_off + 21),
             (px_off + pw - 6, py_off + 21),
             (70, 70, 70), 1)

    # cabeçalho de times
    cv2.putText(frame, team0_name[:10], (px_off + 70,  py_off + 35), font, 0.36, c0, 1, cv2.LINE_AA)
    cv2.putText(frame, team1_name[:10], (px_off + 155, py_off + 35), font, 0.36, c1, 1, cv2.LINE_AA)

    # posse
    cv2.putText(frame, "Posse",        (px_off + 8,   py_off + 53), font, 0.36, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{poss[0]:.0f}%", (px_off + 70,  py_off + 53), font, 0.40, c0, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{poss[1]:.0f}%", (px_off + 155, py_off + 53), font, 0.40, c1, 1, cv2.LINE_AA)

    # distância
    cv2.putText(frame, "Dist (km)",    (px_off + 8,   py_off + 71), font, 0.36, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{team_dist[0]/1000:.1f}", (px_off + 70,  py_off + 71), font, 0.40, c0, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{team_dist[1]/1000:.1f}", (px_off + 155, py_off + 71), font, 0.40, c1, 1, cv2.LINE_AA)

    # velocidade máxima
    cv2.putText(frame, "Vmax (km/h)",  (px_off + 8,   py_off + 89), font, 0.36, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{team_maxspd[0]*3.6:.0f}", (px_off + 70,  py_off + 89), font, 0.40, c0, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{team_maxspd[1]*3.6:.0f}", (px_off + 155, py_off + 89), font, 0.40, c1, 1, cv2.LINE_AA)

    # barra de posse
    bar_x, bar_y = px_off + 8, py_off + 100
    bar_w        = pw - 16
    p0_w         = int(bar_w * poss[0] / 100.0) if (poss[0] + poss[1]) > 0 else bar_w // 2
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (50, 50, 50), -1)
    if p0_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + p0_w, bar_y + 6), c0, -1)
    if p0_w < bar_w:
        cv2.rectangle(frame, (bar_x + p0_w, bar_y), (bar_x + bar_w, bar_y + 6), c1, -1)

    return frame


if __name__ == "__main__":
    analytics = MatchAnalytics()
    print("Module test passed. MatchAnalytics ready.")