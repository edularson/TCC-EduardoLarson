import yaml
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm

# Local modules
from modules.detection import FootballDetector
from modules.tracking import FootballTracker
from modules.field_detection import FieldDetector
from modules.homography import PitchMapper
from modules.shot_detection import ShotDetector
from modules.xg_model import XGModel
from modules.team_classifier import TeamClassifierWrapper
from modules.utils import (
    draw_detections_on_frame,
    draw_shot_event,
    create_radar_view,
    draw_pitch_with_xg
)


class VarzeaVisionPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        print("=" * 60)
        print("Initializing VarzeaVision Pipeline...")
        print("=" * 60)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.detector = FootballDetector(config_path)
        self.tracker = FootballTracker(config_path)
        self.field_detector = FieldDetector(config_path)
        self.pitch_mapper = PitchMapper(config_path)
        self.shot_detector = ShotDetector(config_path)
        self.xg_model = XGModel(config_path)
        self.team_classifier = TeamClassifierWrapper(config_path)

        self.current_clip = None
        self.transformer = None
        self.frame_count = 0
        self.shot_events = []
        self.tracker_id_to_team = {}
        self.ball_positions = {}

        print("Pipeline initialized successfully!\n")

    def _create_sv_detections(self, balls, players, goalkeepers, referees):
        """Convert detector outputs to supervision Detections object."""
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for ball in balls:
            x1, y1, x2, y2, conf = ball
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)
            all_class_ids.append(0)

        for player in players:
            x1, y1, x2, y2, conf, cls_id = player
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)
            all_class_ids.append(cls_id)

        for gk in goalkeepers:
            x1, y1, x2, y2, conf, cls_id = gk
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)
            all_class_ids.append(cls_id)

        for ref in referees:
            x1, y1, x2, y2, conf, cls_id = ref
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)
            all_class_ids.append(cls_id)

        if len(all_boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(all_boxes, dtype=float),
            confidence=np.array(all_scores),
            class_id=np.array(all_class_ids)
        )

    def _get_player_crops(self, frame, detections):
        """Extract player crops for team classification."""
        crops = []
        track_ids = []
        for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
            if track_id is None:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                track_ids.append(int(track_id))
        return crops, track_ids

    def process_clip(self, video_path: str, output_path: str = None):
        """Process a single tactical cam clip end-to-end."""
        print(f"\n{'=' * 60}")
        print(f"Processing clip: {video_path}")
        print(f"{'=' * 60}\n")

        self.current_clip = video_path
        self.shot_events = []
        self.frame_count = 0
        self.tracker_id_to_team = {}
        self.ball_positions = {}

        self.tracker.reset()
        self.pitch_mapper.reset_clip()
        self.shot_detector.reset_clip()
        self.transformer = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

        if output_path is None:
            output_path = f"outputs/processed_{Path(video_path).stem}.mp4"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        pbar = tqdm(total=total_frames, desc="Processing frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame_time = self.frame_count / fps

            # 1. Detection
            balls, players, goalkeepers, referees = self.detector.get_detections(frame)

            # 2. Tracking
            sv_detections = self._create_sv_detections(balls, players, goalkeepers, referees)
            if len(sv_detections) > 0:
                sv_detections = self.tracker.update(frame, sv_detections)

            # 3. Team classification (a cada 30 frames para performance)
            if self.frame_count % 30 == 0:
                if len(sv_detections) > 0 and sv_detections.tracker_id is not None:
                    player_crops, track_ids = self._get_player_crops(frame, sv_detections)
                    if len(player_crops) > 0:
                        predictions = self.team_classifier.predict(player_crops)
                        for track_id, pred in zip(track_ids, predictions):
                            self.tracker_id_to_team[track_id] = pred

            # 4. Homography (computa uma vez por clip)
            if self.transformer is None:
                try:
                    frame_points, pitch_points, valid_mask = self.field_detector.get_keypoints(frame)
                    if len(frame_points) >= 4:
                        self.transformer = self.pitch_mapper.compute_homography_from_points(
                            frame_points, pitch_points
                        )
                        print(f"Homography computed! Using {len(frame_points)} keypoints.")
                except Exception as e:
                    print(f"Homography failed: {e}")

            # 5. Shot detection
            if len(balls) > 0 and self.transformer is not None:
                ball = balls[0]
                ball_center = self.detector.get_ball_center(ball)
                ball_pos_real = self.pitch_mapper.frame_to_pitch(
                    self.transformer, np.array([ball_center])
                )[0]
                
                # DEBUG - remova depois de confirmar
                if self.frame_count <= 10:
                    print(f"Frame {self.frame_count} | ball_center={ball_center} | ball_pitch={ball_pos_real}")

                self.ball_positions[frame_time] = ball_pos_real

                accel_mag, accel_vec = self.shot_detector.compute_ball_acceleration(
                    ball_pos_real, frame_time
                )

                if len(sv_detections) > 0 and sv_detections.tracker_id is not None:
                    ball_x, ball_y = ball_center
                    closest_player = None
                    closest_dist = float('inf')

                    for xyxy, class_id, track_id in zip(
                        sv_detections.xyxy, sv_detections.class_id, sv_detections.tracker_id
                    ):
                        if track_id is None:
                            continue
                        if class_id != 2:  # Apenas jogadores
                            continue

                        px1, py1, px2, py2 = map(int, xyxy)
                        player_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                        dist = np.sqrt((ball_x - player_center[0])**2 + (ball_y - player_center[1])**2)

                        if dist < closest_dist and dist < 150:
                            closest_dist = dist
                            closest_player = {
                                'track_id': int(track_id),
                                'bbox': (px1, py1, px2, py2),
                                'team': self.tracker_id_to_team.get(int(track_id), -1)
                            }

                    if closest_player is not None:
                        is_shot, shot_data = self.shot_detector.detect_shot(
                            ball_pos_real, frame_time, accel_mag, closest_player
                        )

                        if is_shot:
                            team_id = closest_player['team']

                            # Compute xG usando extract_features corretamente
                            xg_value = None
                            if self.xg_model.model is not None:
                                shot_pos = np.array(ball_pos_real)
                                feat = self.xg_model.extract_features(
                                    shot_pos, body_part=1, def_pressure=0
                                )
                                # Features extras com valores default
                                extra = np.array([0, 0, 0, 0, 0, 0])
                                feat_full = np.append(feat, extra).reshape(1, -1)
                                xg_value = float(self.xg_model.predict_xg(feat_full)[0])

                            shot_event = {
                                'time': frame_time,
                                'frame': self.frame_count,
                                'player_id': closest_player['track_id'],
                                'team': team_id,
                                'position': ball_pos_real.tolist(),
                                'velocity': float(accel_mag),
                                'xg': xg_value
                            }
                            self.shot_events.append(shot_event)

                            if xg_value is not None:
                                print(f"\nShot detected at {frame_time:.1f}s! xG: {xg_value:.3f}")
                            else:
                                print(f"\nShot detected at {frame_time:.1f}s!")

                            shot_tuple = (closest_player['track_id'], 0.0, "detected")
                            frame = draw_shot_event(frame, shot_tuple, xg_value)

                # Mostra posição da bola no frame
                h, w = frame.shape[:2]
                pos_text = f"Ball: ({ball_pos_real[0]:.1f}, {ball_pos_real[1]:.1f})"
                cv2.putText(frame, pos_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, pos_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 6. Anotação visual das detecções
            if len(sv_detections) > 0 and sv_detections.tracker_id is not None:
                detections_dict = {'balls': [], 'players': [], 'goalkeepers': [], 'referees': []}

                for xyxy, conf, cls_id, track_id in zip(
                    sv_detections.xyxy, sv_detections.confidence,
                    sv_detections.class_id, sv_detections.tracker_id
                ):
                    x1, y1, x2, y2 = map(int, xyxy)
                    track_id_int = int(track_id) if track_id is not None else -1
                    team_id = self.tracker_id_to_team.get(track_id_int, -1)

                    if cls_id == 0:
                        detections_dict['balls'].append((x1, y1, x2, y2, float(conf)))
                    elif cls_id == 2:
                        detections_dict['players'].append((x1, y1, x2, y2, float(conf), int(cls_id), track_id_int, team_id))
                    elif cls_id == 1:
                        detections_dict['goalkeepers'].append((x1, y1, x2, y2, float(conf), int(cls_id), track_id_int, team_id))
                    elif cls_id == 3:
                        detections_dict['referees'].append((x1, y1, x2, y2, float(conf), int(cls_id), track_id_int))

                frame = draw_detections_on_frame(frame, detections_dict)

            writer.write(frame)
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        print(f"\nProcessing complete! Output: {output_path}")
        print(f"Detected {len(self.shot_events)} shot events")

        return self.shot_events

    def train_xg_model(self, statsbomb_path: str):
        """Train xG model using StatsBomb data."""
        print(f"\nTraining xG model using StatsBomb data: {statsbomb_path}")
        X_train, y_train = self.xg_model.load_statsbomb_data(statsbomb_path)
        self.xg_model.train(X_train, y_train)
        self.xg_model.save_model()
        print("xG model trained and saved!")

    def train_team_classifier(self, player_crops_dir: str):
        """Train team classifier using player crops directory."""
        print(f"\nTraining team classifier using crops from: {player_crops_dir}")
        self.team_classifier.train(player_crops_dir)
        print("Team classifier trained and saved!")

    def auto_train_team_classifier(self, video_path: str, stride: int = 60):
        """Coleta crops do vídeo e treina o team classifier automaticamente."""
        print("Auto-training team classifier from video...")
        cap = cv2.VideoCapture(video_path)
        crops = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                balls, players, goalkeepers, referees = self.detector.get_detections(frame)
                for det in players + goalkeepers:
                    x1, y1, x2, y2, conf, cls_id = det
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops.append(crop)
            frame_idx += 1

        cap.release()
        print(f"Collected {len(crops)} crops")
        self.team_classifier.train(crops)

'''
if __name__ == "__main__":
    pipeline = VarzeaVisionPipeline()
    pipeline.train_xg_model("data/statsbomb_shots_laliga.csv")
    pipeline.auto_train_team_classifier("data/LIVERPOOLxREAL2022teste.mp4")
    events = pipeline.process_clip("data/LIVERPOOLxREAL2022teste.mp4")
    print(f"Shot events: {events}")
'''

if __name__ == "__main__":
    pipeline = VarzeaVisionPipeline()
    pipeline.train_xg_model("data/statsbomb_shots_laliga.csv")
    # Só treina se não existir classifier salvo
    if not Path("outputs/team_classifier.joblib").exists():
        pipeline.auto_train_team_classifier("data/LIVERPOOLxREAL2022teste.mp4")
    events = pipeline.process_clip("data/LIVERPOOLxREAL2022teste.mp4")
    print(f"Shot events: {events}")