"""
field_detection.py — VarzeaVision
Integração com PnLCalib (CVPR/CVIU 2024-2026) para detecção robusta
de keypoints e cálculo de homografia usando modelo HRNet treinado no SoccerNet.

PnLCalib é uma evolução do No-Bells-Just-Whistles com refinamento PnL não-linear.
"""

import sys
import logging
import numpy as np
import cv2
import torch
import yaml
import torchvision.transforms as T
import torchvision.transforms.functional as f
from pathlib import Path
from PIL import Image

from modules.common import load_config

logger = logging.getLogger(__name__)

PNLCALIB_PATH = Path(__file__).parent.parent.parent / "external" / "PnLCalib"
NBJW_PATH = Path(__file__).parent.parent.parent / "external" / "no_bells_just_whistles"

sys.path.insert(0, str(PNLCALIB_PATH))
sys.path.insert(0, str(PNLCALIB_PATH / "utils"))
sys.path.insert(0, str(PNLCALIB_PATH / "model"))

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)


class FieldDetector:
    """
    Detecta a homografia do campo usando PnLCalib.
    Retorna H (3x3) mapeando coordenadas do campo (metros) → pixels.
    """

    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    def __init__(self, config_path=None):
        self.config = load_config(config_path)

        device_cfg = self.config.get("device", "cpu")
        if device_cfg == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        logger.info(f"FieldDetector device: {self.device}")

        fd_cfg = self.config["models"]["field_detection"]
        weights_kp = fd_cfg.get("weights_kp")
        weights_line = fd_cfg.get("weights_line")

        if not weights_kp or not weights_line:
            raise ValueError(
                "config.yaml precisa ter models.field_detection.weights_kp "
                "e models.field_detection.weights_line"
            )

        use_wp_calib = fd_cfg.get("use_wp_calib", True)

        if use_wp_calib:
            from utils.utils_calib_wp import FramebyFrameCalib as _FramebyFrameCalib
            logger.info("Using PnLCalib (wp) calibration")
        else:
            from utils.utils_calib import FramebyFrameCalib as _FramebyFrameCalib
            logger.info("Using NBJW calibration")

        cfg_path = PNLCALIB_PATH / "config" / "hrnetv2_w48.yaml"
        cfg_l_path = PNLCALIB_PATH / "config" / "hrnetv2_w48_l.yaml"

        cfg = yaml.safe_load(open(cfg_path))
        cfg_l = yaml.safe_load(open(cfg_l_path))

        actual_device = self._try_load_models(cfg, cfg_l, weights_kp, weights_line)

        self.transform = T.Resize((540, 960))
        self.kp_threshold = fd_cfg.get("keypoint_confidence_threshold", 0.15)
        self.line_threshold = fd_cfg.get("line_threshold", 0.40)
        self.pnl_refine = fd_cfg.get("pnl_refine", True)
        self.use_wp_calib = use_wp_calib
        self._FramebyFrameCalib = _FramebyFrameCalib

        self._cam = None
        self._frame_w = 0
        self._frame_h = 0

        logger.info(f"FieldDetector (PnLCalib) initialized on {actual_device}. pnl_refine={self.pnl_refine}")

    def _try_load_models(self, cfg, cfg_l, weights_kp, weights_line):
        for device in [self.device, torch.device("cpu")]:
            try:
                logger.info(f"Attempting to load models on {device}...")
                state = torch.load(weights_kp, map_location=device)
                self.model = get_cls_net(cfg)
                self.model.load_state_dict(state)
                self.model.to(device).eval()

                state_l = torch.load(weights_line, map_location=device)
                self.model_l = get_cls_net_l(cfg_l)
                self.model_l.load_state_dict(state_l)
                self.model_l.to(device).eval()

                if device != self.device:
                    logger.warning(f"MPS loading failed. Fell back to CPU.")
                logger.info(f"Models loaded successfully on {device}.")
                self.device = device
                return device
            except Exception as e:
                logger.warning(f"Failed to load on {device}: {e}")
                self.model = None
                self.model_l = None
                continue
        raise RuntimeError(
            f"Failed to load PnLCalib models from {weights_kp} and {weights_line} "
            f"on MPS, CUDA, or CPU."
        )

    def reset_clip(self):
        self._cam = None
        logger.info("FieldDetector reset for new clip.")

    def get_keypoints(self, frame: np.ndarray) -> tuple:
        H, quality = self.get_homography(frame)
        if H is None:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty, 0.0

        corners_pitch = np.array([
            [0., 0.],
            [self.PITCH_LENGTH, 0.],
            [self.PITCH_LENGTH, self.PITCH_WIDTH],
            [0., self.PITCH_WIDTH],
        ], dtype=np.float64)

        corners_frame = cv2.perspectiveTransform(
            corners_pitch.reshape(-1, 1, 2), np.linalg.inv(H)
        ).reshape(-1, 2)

        return corners_frame, corners_pitch, quality

    def get_homography(self, frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]
        if self._cam is None or self._frame_w != w or self._frame_h != h:
            self._cam = self._FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
            self._frame_w = w
            self._frame_h = h

        try:
            params = self._run_inference(frame)
        except Exception as e:
            logger.warning(f"PnLCalib inference failed: {e}")
            return None, 0.0

        if params is None:
            return None, 0.0

        P = self._build_P(params)
        H_f2p = self._P_to_H(P)
        quality = self._estimate_quality(H_f2p, frame)

        try:
            H_p2f = np.linalg.inv(H_f2p)
            if abs(H_p2f[2, 2]) > 1e-8:
                H_p2f = H_p2f / H_p2f[2, 2]
        except np.linalg.LinAlgError:
            return None, 0.0

        return H_p2f, quality

    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, quality: float = None) -> np.ndarray:
        annotated = frame.copy()
        if len(keypoints) == 0:
            return annotated

        color = (0, 220, 0) if (quality or 0) >= 0.7 else (0, 165, 255) if (quality or 0) >= 0.4 else (0, 0, 220)

        for i, (x, y) in enumerate(keypoints):
            cv2.circle(annotated, (int(x), int(y)), 6, color, -1)
            cv2.putText(annotated, f"K{i}", (int(x) + 8, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if quality is not None:
            label = f"PnLCalib quality: {quality:.2f}"
            cv2.putText(annotated, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return annotated

    def _run_inference(self, frame: np.ndarray) -> dict | None:
        if self.model is None or self.model_l is None:
            logger.error("Models not loaded. Cannot run inference.")
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        tensor = f.to_tensor(img).float().unsqueeze(0)

        if tensor.size()[-1] != 960:
            tensor = self.transform(tensor)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            heatmaps = self.model(tensor)
            heatmaps_l = self.model_l(tensor)

        _, _, h, w = tensor.size()
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_list = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_list = coords_to_dict(line_coords, threshold=self.line_threshold)

        kp_dict_0   = kp_list[0]
        lines_dict_0 = lines_list[0] if lines_list else {}
        kp_completed, lines_completed = complete_keypoints(
            kp_dict_0, lines_dict_0, w=w, h=h, normalize=True
        )

        kp_count   = sum(1 for k in kp_completed if k <= 57)
        line_count = len(lines_completed)
        logger.info(
            f"[PnLCalib] kp_detected={kp_count}, lines_detected={line_count}, "
            f"frame_size={w}x{h}, threshold_kp={self.kp_threshold}, threshold_line={self.line_threshold}"
        )

        # ── calibração inicial ──────────────────────────────────────────────────────
        if not hasattr(self._cam, 'intrinsics') or self._cam.intrinsics is None:
            fx = fy = float(max(self._frame_w, self._frame_h))
            K    = np.array([[fx, 0., self._frame_w / 2.],
                            [0., fy, self._frame_h / 2.],
                            [0.,  0.,              1.  ]], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
        else:
            K    = self._cam.intrinsics
            dist = self._cam.distortion

        calibration = {"intrinsics": K, "distortion": dist}
        self._cam.update(calibration, kp_completed, lines_completed)
        # ────────────────────────────────────────────────────────────────────────────

        if self._cam.intrinsics is None:
            H_init, _ = self._cam.get_homography_from_ground_plane(
                use_ransac=10, inverse=False, refine_lines=False
            )
            if H_init is not None:
                self._cam.estimate_calibration_matrix_from_plane_homography(H_init)

        params = self._cam.heuristic_voting(refine_lines=self.pnl_refine)

        if params is None:
            cam_params, ret = self._cam.get_cam_params(mode="full", use_ransac=15, refine=False, refine_w_lines=False)
            if cam_params is not None and ret:
                params = {"cam_params": cam_params, "mode": "full_fallback"}
            if params is None:
                cam_params, ret = self._cam.get_cam_params(mode="ground_plane", use_ransac=15, refine=False, refine_w_lines=False)
                if cam_params is not None and ret:
                    params = {"cam_params": cam_params, "mode": "ground_plane_fallback"}

        return params

    def _build_P(self, params: dict) -> np.ndarray:
        cam = params["cam_params"]

        fl_x = float(cam["x_focal_length"])
        fl_y = float(cam["y_focal_length"])
        px, py = cam["principal_point"]

        K = np.array([
            [fl_x, 0., float(px)],
            [0., fl_y, float(py)],
            [0., 0., 1.],
        ], dtype=np.float64)

        R = np.array(cam["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        pos = np.array(cam["position_meters"], dtype=np.float64).reshape(3, 1)
        t = -R @ pos

        Rt = np.hstack([R, t])
        P = K @ Rt

        return P

    def _P_to_H(self, P: np.ndarray) -> np.ndarray:
        H = P[:, [0, 1, 3]].copy().astype(np.float64)
        if abs(H[2, 2]) > 1e-8:
            H = H / H[2, 2]
        return H

    def _estimate_quality(self, H: np.ndarray, frame: np.ndarray) -> float:
        if H is None:
            return 0.0

        h, w = frame.shape[:2]

        corners = np.array([
            [0.0, 0.0, 1.0],
            [self.PITCH_LENGTH, 0.0, 1.0],
            [self.PITCH_LENGTH, self.PITCH_WIDTH, 1.0],
            [0.0, self.PITCH_WIDTH, 1.0],
        ], dtype=np.float64)

        projected = []
        for c in corners:
            p = H @ c
            if abs(p[2]) > 1e-8:
                p = p / p[2]
            projected.append(p[:2])

        projected = np.array(projected)

        margin = 0.3
        in_frame = (
            (projected[:, 0] > -w * margin) &
            (projected[:, 0] < w * (1 + margin)) &
            (projected[:, 1] > -h * margin) &
            (projected[:, 1] < h * (1 + margin))
        )

        return float(in_frame.mean())


if __name__ == "__main__":
    import cv2

    detector = FieldDetector()

    caminho_img = "src/data/seu_frame.jpg"
    frame = cv2.imread(caminho_img)

    if frame is None:
        print(f"Imagem não encontrada: {caminho_img}")
    else:
        H, quality = detector.get_homography(frame)
        print(f"quality: {quality:.3f}")

        if H is not None:
            H_f2p = np.linalg.inv(H)

            corners = np.array([
                [0.0, 0.0],
                [105.0, 0.0],
                [105.0, 68.0],
                [0.0, 68.0],
            ], dtype=np.float64)

            for pt in corners:
                p = H_f2p @ np.array([pt[0], pt[1], 1.0])
                if abs(p[2]) > 1e-8:
                    p = p / p[2]
                    x, y = int(p[0]), int(p[1])

                    if -10000 < x < 10000 and -10000 < y < 10000:
                        cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)

            centro = H_f2p @ np.array([52.5, 34.0, 1.0])
            if abs(centro[2]) > 1e-8:
                centro = centro / centro[2]
                cv2.circle(frame, (int(centro[0]), int(centro[1])), 15, (0, 255, 0), -1)

        cv2.imwrite("src/data/homografia_teste.jpg", frame)
        print("Salvo em src/data/homografia_teste.jpg")