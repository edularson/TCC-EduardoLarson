"""
field_detection.py — VarzeaVision
Detecção robusta de keypoints do campo com validação por reprojection error.

Melhorias em relação à versão anterior:
1. Validação por reprojection error: rejeita keypoints que não formam uma
   homografia geometricamente coerente (erro > threshold em pixels).
2. Seleção por RANSAC interna via cv2.findHomography(..., cv2.RANSAC):
   elimina outliers automaticamente entre os keypoints detectados.
3. Score de qualidade retornado junto com os pontos, permitindo que o
   PitchMapper decida se aceita ou descarta o frame.
4. Fallback gracioso: se menos de 4 keypoints válidos, retorna array vazio
   com qualidade 0.0 — nunca levanta exceção no pipeline principal.
"""

import yaml
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
from pathlib import Path


class FieldDetector:
    def __init__(self, config_path=None):
        module_dir = Path(__file__).parent
        project_root = module_dir.parent

        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            self.device = 'cpu'
            print("MPS not available, falling back to CPU")

        # Carrega modelo de keypoints treinado
        self.model_path = str(
            project_root / "TCC_VarzeaVision_Keypoint_Backup/datasets/runs/pose/train/weights/best.pt"
        )
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Loaded trained keypoint model: {self.model_path}")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            print("Falling back to yolov8x-pose-p6.pt...")
            self.model = YOLO("yolov8x-pose-p6.pt")
            self.model.to(self.device)

        self.conf_threshold = self.config['models']['field_detection']['confidence_threshold']
        self.keypoint_conf_threshold = self.config['models']['field_detection']['keypoint_confidence_threshold']

        # Threshold de reprojection error em pixels.
        # Valores acima disso indicam que a homografia estimada é ruim.
        # 10px é conservador mas seguro para câmera tática a ~30m de distância.
        self.max_reproj_error_px = self.config['models']['field_detection'].get(
            'max_reproj_error_px', 10.0
        )

        self.config_pitch = SoccerPitchConfiguration()
        # Vértices do campo em metros reais (divididos por 100 pois SoccerPitchConfig usa cm)
        self._pitch_vertices_m = np.array(self.config_pitch.vertices, dtype=np.float64) / 100.0

        print(f"FieldDetector initialized on device: {self.device}")
        print(f"Keypoint conf threshold: {self.keypoint_conf_threshold}")
        print(f"Max reprojection error: {self.max_reproj_error_px}px")

    # ------------------------------------------------------------------
    # Inferência bruta
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray):
        """Roda YOLOv8-pose e retorna o Results object."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=1280,
            device=self.device,
            verbose=False
        )
        return results[0]

    # ------------------------------------------------------------------
    # Extração de keypoints com score de qualidade
    # ------------------------------------------------------------------

    def get_keypoints(self, frame: np.ndarray) -> tuple:
        """
        Extrai keypoints válidos do frame e retorna junto com um score de
        qualidade da homografia resultante.

        Returns:
            frame_points  : (N, 2) float64 — coordenadas em pixels no frame
            pitch_points  : (N, 2) float64 — coordenadas em metros no campo real
            quality_score : float em [0, 1] — 1.0 = homografia perfeita,
                            0.0 = inválida ou sem pontos suficientes

        A função NUNCA levanta exceção — retorna arrays vazios + quality=0.0
        em qualquer condição de falha.
        """
        empty = np.empty((0, 2), dtype=np.float64)

        try:
            result = self.infer(frame)
        except Exception as e:
            print(f"[FieldDetector] Inference failed: {e}")
            return empty, empty, 0.0

        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return empty, empty, 0.0
        if result.keypoints.conf is None or len(result.keypoints.conf) == 0:
            return empty, empty, 0.0

        # Seleciona a detecção com maior confiança média entre os keypoints
        mean_confs = result.keypoints.conf.mean(dim=1)
        best_idx = mean_confs.argmax().item()

        kpts_xy   = result.keypoints.xy[best_idx].cpu().numpy().astype(np.float64)   # (K, 2)
        kpts_conf = result.keypoints.conf[best_idx].cpu().numpy()                    # (K,)

        # Filtra por confiança mínima
        conf_mask = kpts_conf >= self.keypoint_conf_threshold

        # Garante que o índice não excede o número de vértices conhecidos
        n_vertices = len(self._pitch_vertices_m)
        index_mask = np.arange(len(kpts_xy)) < n_vertices

        valid_mask = conf_mask & index_mask
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 4:
            # Sem pontos suficientes para homografia
            return empty, empty, 0.0

        frame_pts = kpts_xy[valid_indices]               # (N, 2)
        pitch_pts = self._pitch_vertices_m[valid_indices] # (N, 2)

        # Valida a qualidade via reprojection error com RANSAC interno
        quality = self._compute_quality(frame_pts, pitch_pts)

        return frame_pts, pitch_pts, quality

    # ------------------------------------------------------------------
    # Validação por Reprojection Error
    # ------------------------------------------------------------------

    def _compute_quality(self, frame_pts: np.ndarray, pitch_pts: np.ndarray) -> float:
        """
        Estima a qualidade dos keypoints calculando uma homografia temporária
        com RANSAC e medindo o reprojection error médio.

        O RANSAC interno do cv2.findHomography já filtra outliers — então o
        erro reportado é sobre os inliers, não sobre todos os pontos.

        Returns:
            float em [0, 1]:
                1.0 → erro médio = 0px (perfeito)
                0.5 → erro médio = max_reproj_error_px / 2
                0.0 → homografia inválida ou erro > max_reproj_error_px
        """
        if len(frame_pts) < 4:
            return 0.0

        try:
            # cv2.findHomography espera shapes (N,1,2) ou (N,2)
            # Mapeia pitch_pts → frame_pts (inverso do que usamos no pipeline,
            # mas só para calcular o erro de reprojeção aqui)
            H, inlier_mask = cv2.findHomography(
                pitch_pts.reshape(-1, 1, 2),
                frame_pts.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.max_reproj_error_px
            )
        except cv2.error:
            return 0.0

        if H is None:
            return 0.0

        inliers = (inlier_mask.ravel() == 1)
        n_inliers = inliers.sum()

        if n_inliers < 4:
            return 0.0

        # Reprojeção: transforma pitch_pts (inliers) de volta para pixels
        pts_pitch_inliers = pitch_pts[inliers].reshape(-1, 1, 2)
        pts_frame_inliers = frame_pts[inliers]

        projected = cv2.perspectiveTransform(pts_pitch_inliers, H)
        projected = projected.reshape(-1, 2)

        errors = np.linalg.norm(projected - pts_frame_inliers, axis=1)
        mean_error = errors.mean()

        # Converte para score 0-1 (erro 0px → 1.0, erro ≥ threshold → 0.0)
        quality = max(0.0, 1.0 - (mean_error / self.max_reproj_error_px))

        # Penaliza levemente se poucos inliers (menos de 6 é instável)
        inlier_ratio = n_inliers / len(frame_pts)
        quality *= min(1.0, inlier_ratio * 1.5)

        return float(quality)

    # ------------------------------------------------------------------
    # Utilitário de visualização (debug)
    # ------------------------------------------------------------------

    def draw_keypoints(self, frame: np.ndarray,
                       keypoints: np.ndarray,
                       quality: float = None) -> np.ndarray:
        """Desenha keypoints no frame com cor indicando a qualidade."""
        annotated = frame.copy()
        if len(keypoints) == 0:
            return annotated

        # Verde = boa qualidade, Vermelho = ruim
        if quality is None:
            color = (0, 165, 255)  # laranja padrão
        elif quality >= 0.7:
            color = (0, 220, 0)    # verde
        elif quality >= 0.4:
            color = (0, 165, 255)  # laranja
        else:
            color = (0, 0, 220)    # vermelho

        for i, (x, y) in enumerate(keypoints):
            cv2.circle(annotated, (int(x), int(y)), 6, color, -1)
            cv2.putText(annotated, f"K{i}", (int(x) + 8, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if quality is not None:
            label = f"KP quality: {quality:.2f} ({len(keypoints)} pts)"
            cv2.putText(annotated, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return annotated


if __name__ == "__main__":
    detector = FieldDetector()
    print("Module test passed. FieldDetector ready.")