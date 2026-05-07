import cv2
import numpy as np
from modules.field_detection import FieldDetector
from modules.homography import PitchMapper
from sports.configs.soccer import SoccerPitchConfiguration

# Carrega primeiro frame do vídeo
cap = cv2.VideoCapture("data/LIVERPOOLxREAL2022teste.mp4")
ret, frame = cap.read()
cap.release()

detector = FieldDetector()
mapper = PitchMapper()
config = SoccerPitchConfiguration()

# Detecta keypoints
frame_points, pitch_points, valid_mask = detector.get_keypoints(frame)
print(f"Keypoints válidos: {len(frame_points)}")

# Desenha keypoints no frame com índice e coordenada real
debug_frame = frame.copy()
for i, (fp, pp) in enumerate(zip(frame_points, pitch_points)):
    x, y = int(fp[0]), int(fp[1])
    px, py = pp[0]/100, pp[1]/100  # cm -> metros
    cv2.circle(debug_frame, (x, y), 8, (0, 255, 0), -1)
    cv2.putText(debug_frame, f"{px:.0f},{py:.0f}m", (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Testa alguns pontos conhecidos do campo
transformer = mapper.compute_homography_from_points(frame_points, pitch_points)

# Pontos de referência para testar
test_points = {
    "centro_frame": (frame.shape[1]//2, frame.shape[0]//2),
}

for name, pt in test_points.items():
    real = mapper.frame_to_pitch(transformer, np.array([pt]))[0]
    print(f"{name}: pixel={pt} -> pitch=({real[0]/100:.1f}m, {real[1]/100:.1f}m)")
    cv2.circle(debug_frame, pt, 10, (0, 0, 255), -1)
    cv2.putText(debug_frame, f"{real[0]/100:.0f},{real[1]/100:.0f}m", 
                (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite("outputs/debug_homography.png", debug_frame)
print("Salvo em outputs/debug_homography.png")