import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Importando a nossa Camada de Configuração Otimizada (Memória Cache)
from modules.common import load_config

class FootballDetector:
    def __init__(self, config_path=None):
        # I/O zero-latency: Carrega direto da memória
        self.config = load_config(config_path)
        
        # Set device (MPS for M4 Pro, fallback to CPU)
        self.device = self.config['device']
        if self.device == 'mps' and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = 'cpu'
        
        # Load detection model (local YOLOv8 weights)
        model_path = self.config['models']['detection']['path']
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Class mapping (confirmed: ball:0, goalkeeper:1, player:2, referee:3)
        self.class_ids = self.config['models']['detection']['class_ids']
        self.conf_threshold = self.config['models']['detection']['confidence_threshold']
        self.nms_threshold = self.config['models']['detection']['nms_threshold']
        self.imgsz = self.config['models']['detection']['imgsz']
        
        print(f"FootballDetector initialized on device: {self.device}")
        print(f"Model: {model_path}, Classes: {self.config['models']['detection']['class_names']}")

    def infer(self, frame: np.ndarray) -> any:
        """
        Run YOLOv8 inference on a single frame.
        Returns Ultralytics Results object.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        return results[0]  # Single frame, take first result

    def get_detections(self, frame: np.ndarray) -> tuple:
        """
        Extrai detecções de forma vetorizada, mitigando o overhead de sincronização de tensores.
        Retorna arrays NumPy no formato (N, 5) para bolas e (N, 6) para os demais,
        onde as colunas são: [x1, y1, x2, y2, conf, (opcional: cls_id)].
        """
        result = self.infer(frame)
        if result.boxes is None or len(result.boxes.cls) == 0:
            return np.empty((0, 5)), np.empty((0, 6)), np.empty((0, 6)), np.empty((0, 6))
        
        # Sincronização única GPU -> CPU (Máxima Performance)
        boxes_data = result.boxes.data.cpu().numpy() # [N, 6] -> [x1, y1, x2, y2, conf, cls]
        
        classes = boxes_data[:, 5].astype(int)
        
        # Máscaras booleanas (Operação O(N) em C, sem loops em Python)
        mask_ball = (classes == self.class_ids['ball'])
        mask_player = (classes == self.class_ids['player'])
        mask_gk = (classes == self.class_ids['goalkeeper'])
        mask_ref = (classes == self.class_ids['referee'])
        
        # Bola não precisa do cls_id no retorno segundo a sua assinatura
        balls = boxes_data[mask_ball][:, :5] 
        players = boxes_data[mask_player]
        goalkeepers = boxes_data[mask_gk]
        referees = boxes_data[mask_ref]
        
        return balls, players, goalkeepers, referees


    def get_ball_center(self, ball_detection: tuple) -> tuple:
        """Get center (x,y) of ball bbox."""
        x1, y1, x2, y2, _ = ball_detection
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_player_foot_position(self, player_detection: tuple) -> tuple:
        """Get bottom-center (foot position) of player bbox."""
        x1, y1, x2, y2, *_ = player_detection
        return ((x1 + x2) // 2, y2)


if __name__ == "__main__":
    # Test detection module
    detector = FootballDetector()
    print("Module test passed. Detector ready.")