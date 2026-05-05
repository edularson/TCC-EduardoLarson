# debug_keypoints.py
from modules.field_detection import FieldDetector
import cv2

detector = FieldDetector()
cap = cv2.VideoCapture("data/LIVERPOOLxREAL2022.mp4")
ret, frame = cap.read()
cap.release()

fp, pp, mask = detector.get_keypoints(frame)
print(f"Keypoints válidos: {len(fp)}")
print(f"Frame points:\n{fp}")