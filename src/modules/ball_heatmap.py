"""
ball_heatmap.py — VarzeaVision
Módulo de refinamento de posição da bola via regressão de heatmap gaussiano.

Arquitetura: U-Net com encoder MobileNetV2 pré-treinado.
Entrada:  patch RGB 128x128 centrado na bbox da bola (do YOLO ou do Kalman)
Saída:    (cx, cy, confidence) em coordenadas do frame original
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from modules.common import load_config, get_project_root


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        mb       = models.mobilenet_v2(weights=None)
        features = mb.features

        self.enc1   = features[0:2]
        self.enc2   = features[2:4]
        self.enc3   = features[4:7]
        self.enc4   = features[7:14]
        self.bridge = features[14:]

        self.up4  = nn.ConvTranspose2d(1280, 96, 2, stride=2)
        self.dec4 = ConvBnRelu(96 + 96, 96)

        self.up3  = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.dec3 = ConvBnRelu(32 + 32, 32)

        self.up2  = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec2 = ConvBnRelu(24 + 24, 24)

        self.up1  = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec1 = ConvBnRelu(16 + 16, 16)

        self.up0  = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec0 = ConvBnRelu(8, 8)

        self.head = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bridge(e4)

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(self.up0(d1))

        return self.head(d0)


class BallHeatmapRefiner:
    """
    Refina a posição da bola detectada pelo YOLO usando heatmap gaussiano.
    Quando o YOLO não detecta a bola, usa a última posição conhecida como ROI.
    """

    PATCH_SIZE = 128

    def __init__(self, config_path=None):
        self.config = load_config(config_path)

        weights_path = get_project_root() / 'runs' / 'ball_heatmap_best.pt'

        if not weights_path.exists():
            print(f'[BallHeatmap] Pesos não encontrados em {weights_path}. Módulo desativado.')
            self.model  = None
            self.device = None
            return

        # Device — MPS no Mac, CPU fallback
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = BallUNet().to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self._last_bbox        = None
        self._frames_no_yolo   = 0
        self._MAX_FALLBACK     = 5
        print(f'[BallHeatmap] Modelo carregado. Device: {self.device}')

    def is_available(self) -> bool:
        return self.model is not None

    def reset(self):
        self._last_bbox      = None
        self._frames_no_yolo = 0

    def refine(self,
               frame: np.ndarray,
               yolo_bbox: np.ndarray | None) -> np.ndarray | None:
        """
        Refina a posição da bola.

        Args:
            frame:     frame BGR completo.
            yolo_bbox: array (5,) [x1, y1, x2, y2, conf] do YOLO.
                       None quando o YOLO não detectou a bola.

        Returns:
            Array (5,) [x1, y1, x2, y2, conf] com posição refinada,
            ou None se não for possível estimar.
        """
        if self.model is None:
            return yolo_bbox

        # Determina a ROI — bbox do YOLO ou fallback limitado
        if yolo_bbox is not None:
            roi_bbox             = yolo_bbox
            self._last_bbox      = yolo_bbox.copy()
            self._frames_no_yolo = 0
        elif self._last_bbox is not None and self._frames_no_yolo < self._MAX_FALLBACK:
            roi_bbox              = self._last_bbox
            self._frames_no_yolo += 1
        else:
            self._frames_no_yolo += 1
            return None

        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = roi_bbox[:4]
        cx_roi = (x1 + x2) / 2
        cy_roi = (y1 + y2) / 2

        # Recorta patch 128x128 centrado na bbox
        half = self.PATCH_SIZE // 2
        rx1  = int(max(0, cx_roi - half))
        ry1  = int(max(0, cy_roi - half))
        rx2  = min(img_w, rx1 + self.PATCH_SIZE)
        ry2  = min(img_h, ry1 + self.PATCH_SIZE)

        patch = frame[ry1:ry2, rx1:rx2]
        if patch.shape[0] < 16 or patch.shape[1] < 16:
            return yolo_bbox

        patch_resized = cv2.resize(patch, (self.PATCH_SIZE, self.PATCH_SIZE))
        patch_rgb     = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
        tensor        = torch.from_numpy(
            patch_rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inferência
        with torch.no_grad():
            heatmap = self.model(tensor)[0, 0].cpu().numpy()

        # Extrai centro sub-pixel via argmax
        peak_idx   = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        cy_local   = float(peak_idx[0])
        cx_local   = float(peak_idx[1])
        confidence = float(heatmap[peak_idx])

        # Confiança baixa — heatmap não convergiu, retorna detecção original
        if confidence < 0.3:
            return yolo_bbox

        # Converte de coordenadas do patch para coordenadas do frame
        scale_x    = (rx2 - rx1) / self.PATCH_SIZE
        scale_y    = (ry2 - ry1) / self.PATCH_SIZE
        cx_global  = rx1 + cx_local * scale_x
        cy_global  = ry1 + cy_local * scale_y

        # Reconstrói bbox com mesmo tamanho da original mas centro refinado
        bw = (x2 - x1) / 2
        bh = (y2 - y1) / 2

        refined = np.array([
            cx_global - bw,
            cy_global - bh,
            cx_global + bw,
            cy_global + bh,
            confidence,
        ], dtype=np.float32)

        # Atualiza última bbox válida
        self._last_bbox = refined.copy()

        return refined
