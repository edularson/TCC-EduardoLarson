import cv2
import numpy as np
from modules.field_detection import FieldDetector
from sports.configs.soccer import SoccerPitchConfiguration

cap = cv2.VideoCapture("data/LIVERPOOLxREAL2022teste.mp4")
ret, frame = cap.read()
cap.release()

detector = FieldDetector()
config = SoccerPitchConfiguration()
result = detector.infer(frame)

# Pega a melhor detecção
mean_confs = result.keypoints.conf.mean(dim=1)
best_idx = mean_confs.argmax().item()

kpts = result.keypoints.xy[best_idx].cpu().numpy()
kpts_conf = result.keypoints.conf[best_idx].cpu().numpy()
vertices = np.array(config.vertices) / 100.0  # cm -> metros

print("Índice | Conf  | Pixel (x,y)      | Vertex (x,y)m")
print("-" * 60)
for i in range(32):
    conf = kpts_conf[i]
    px, py = kpts[i]
    vx, vy = vertices[i]
    flag = "✓" if conf >= 0.5 else " "
    print(f"  {i:2d}   | {conf:.2f} | ({px:6.0f}, {py:6.0f}) | ({vx:.1f}, {vy:.1f})m  {flag}")