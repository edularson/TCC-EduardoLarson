"""
debug_homography.py — Debug visual da homografia do campo

Desenha as linhas do campo sobre o vídeo com cores indicando a qualidade da homografia:
- Verde: qualidade >= 0.7 (boa)
- Laranja: 0.3 <= qualidade < 0.7 (média)
- Vermelho: qualidade < 0.3 (ruim/inválida)

Controles:
- SPACE: pause/resume
- Q: sair
"""
import sys
import cv2
import numpy as np
import logging
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "external" / "PnLCalib"))
sys.path.insert(0, str(PROJECT_DIR / "external" / "PnLCalib" / "utils"))
sys.path.insert(0, str(PROJECT_DIR / "external" / "PnLCalib" / "model"))

from modules.field_detection import FieldDetector
from modules.homography import PitchMapper

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def get_quality_color(quality: float) -> tuple:
    """Retorna cor baseada na qualidade da homografia."""
    if quality >= 0.7:
        return (0, 255, 0)
    elif quality >= 0.3:
        return (0, 165, 255)
    else:
        return (0, 0, 255)


def draw_field_lines(img: np.ndarray, H: np.ndarray, color: tuple, thickness: int = 3):
    """Desenha as linhas do campo usando homografia."""
    if H is None:
        return

    field_points = []

    field_points.append(("contorno", [(0, 0), (PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH), (0, PITCH_WIDTH)]))
    field_points.append(("linha_meio", [(52.5, 0), (52.5, PITCH_WIDTH)]))

    field_points.append(("area_grande_esq", [(0, 13.84), (16.5, 54.16)]))
    field_points.append(("area_grande_dir", [(88.5, 13.84), (105, 54.16)]))

    field_points.append(("area_peq_esq", [(0, 24.84), (5.5, 43.16)]))
    field_points.append(("area_peq_dir", [(99.5, 24.84), (105, 43.16)]))

    field_points.append(("ponto_penalti_esq", [(11, 34)]))
    field_points.append(("ponto_penalti_dir", [(94, 34)]))

    center_circle_pts = []
    for angle in np.linspace(0, 2 * np.pi, 60):
        x = 52.5 + 9.15 * np.cos(angle)
        y = 34 + 9.15 * np.sin(angle)
        center_circle_pts.append((x, y))
    field_points.append(("circulo_centro", center_circle_pts))

    mapper = PitchMapper()

    for name, points in field_points:
        if isinstance(points[0], list) and len(points[0]) == 2 and isinstance(points[0][0], (int, float)):
            pts = np.array(points, dtype=np.float32)
        else:
            pts = np.array(points, dtype=np.float32)

        if len(pts) == 1:
            projected = mapper.pitch_to_frame(H, pts)
            if projected is not None and len(projected) > 0:
                cv2.circle(img, (int(projected[0][0]), int(projected[0][1])), 5, color, -1)
        elif len(pts) == 2:
            p1 = mapper.pitch_to_frame(H, pts[:1])[0]
            p2 = mapper.pitch_to_frame(H, pts[1:])[0]
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
        elif len(pts) >= 3:
            projected = mapper.pitch_to_frame(H, pts)
            if projected is not None and len(projected) >= 3:
                pts_int = np.int32(projected.reshape(-1, 1, 2))
                if name == "circulo_centro":
                    for i in range(len(projected)):
                        p1 = projected[i]
                        p2 = projected[(i + 1) % len(projected)]
                        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
                else:
                    cv2.polylines(img, [pts_int], True, color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Debug visual da homografia do campo")
    parser.add_argument("--input", "-i", type=str,
                        default=str(PROJECT_DIR / "src/data/SPAINxCROATIA2.mp4"),
                        help="Caminho do vídeo de entrada")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Frame inicial")
    parser.add_argument("--config", "-c", type=str,
                        default=str(PROJECT_DIR / "src/configs/config.yaml"),
                        help="Arquivo de configuração")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Inicializando FieldDetector e PitchMapper...")
    detector = FieldDetector(config_path=args.config)
    mapper = PitchMapper(config_path=args.config)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Não foi possível abrir o vídeo: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Vídeo: {args.input}")
    logger.info(f"Resolução: {width}x{height}")
    logger.info(f"FPS: {fps}, Total de frames: {total}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    paused = False
    frame_idx = args.start

    logger.info("Controles: SPACE = pause/resume, Q = sair")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                logger.info("Fim do vídeo")
                break
            frame_idx += 1
        else:
            ret = True

        if ret and frame is not None:
            H_raw, quality = detector.get_homography(frame)

            # Bypass no Kalman Filter para ver a detecção pura
            if H_raw is not None:
                H = mapper._nbjw_to_pipeline(H_raw)
            else:
                H = None

            color = get_quality_color(quality)

            if H is not None:
                draw_field_lines(frame, H, color, thickness=3)

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (280, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (280, 90), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Frame: {frame_idx}/{total}", (20, 35),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Qualidade: {quality:.3f}", (20, 55),
                        font, 0.5, color, 1, cv2.LINE_AA)

            status = "PAUSADO" if paused else "EXECUTANDO"
            cv2.putText(frame, f"Status: {status}", (20, 75),
                        font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Debug Homografia", frame)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            logger.info("Saindo...")
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()