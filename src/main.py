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

        # self.transformer agora é np.ndarray (3x3) ou None
        # NÃO é mais ViewTransformer — é a matriz H diretamente
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
        # MUDANÇA: reset do PitchMapper e ShotDetector entre clips
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

            # SMOKE TEST — remover após validar
            if self.frame_count > 150:
                print("[SMOKE TEST] 150 frames processados, encerrando.")
                break

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

            # 4. Homography — MUDANÇA: atualiza todo frame via PitchMapper robusto
            # O PitchMapper decide internamente se usa H nova, média histórica ou Kalman
            frame_pts, pitch_pts, kp_quality = self.field_detector.get_keypoints(frame)
            H = self.pitch_mapper.get_homography(frame_pts, pitch_pts, kp_quality)
            self.transformer = H  # None se ainda sem estimativa válida

            # Log de qualidade a cada 5 segundos
            if self.frame_count % int(fps * 5) == 0:
                if H is not None:
                    reproj_err = self.pitch_mapper.compute_reprojection_error(
                        H, frame_pts, pitch_pts
                    ) if len(frame_pts) >= 4 else float('inf')
                    print(f"  [H] frame={self.frame_count} | "
                          f"quality={kp_quality:.2f} | "
                          f"reproj={reproj_err:.1f}px | "
                          f"kp={len(frame_pts)}")
                else:
                    print(f"  [H] frame={self.frame_count} | SEM HOMOGRAFIA")

            # 5. Shot detection — MUDANÇA: velocidade em vez de aceleração,
            # goal_pos via pitch_mapper (consistente com invert_x_axis)
            if len(balls) > 0 and self.transformer is not None:
                ball = balls[0]
                ball_center = self.detector.get_ball_center(ball)
                ball_pos_real = self.pitch_mapper.frame_to_pitch(
                    self.transformer, np.array([ball_center])
                )[0]

                self.ball_positions[frame_time] = ball_pos_real

                # MUDANÇA: compute_ball_velocity em vez de compute_ball_acceleration
                '''
                speed_mag, vel_vec = self.shot_detector.compute_ball_velocity(
                    ball_pos_real, frame_time
                )
                '''
                ret = self.shot_detector.compute_ball_velocity(ball_pos_real, frame_time)
                
                # Trata caso a função retorne (vetor, magnitude) ou (magnitude, vetor)
                if isinstance(ret[0], np.ndarray) and ret[0].size > 1:
                    vel_vec = ret[0]
                    speed_mag = float(np.linalg.norm(vel_vec)) # Força virar um número único
                else:
                    speed_mag = float(ret[0])
                    vel_vec = ret[1]

                if len(sv_detections) > 0 and sv_detections.tracker_id is not None:
                    ball_x, ball_y = ball_center
                    closest_player = None
                    closest_dist = float('inf')

                    for xyxy, class_id, track_id in zip(
                        sv_detections.xyxy, sv_detections.class_id, sv_detections.tracker_id
                    ):
                        if track_id is None:
                            continue
                        if class_id != 2:  # Apenas jogadores de campo
                            continue

                        px1, py1, px2, py2 = map(int, xyxy)
                        foot_pixel = self.detector.get_player_foot_position(
                            (px1, py1, px2, py2, 0)
                        )
                        foot_pos_real = self.pitch_mapper.frame_to_pitch(
                            self.transformer, np.array([foot_pixel])
                        )[0]

                        dist_real = np.linalg.norm(ball_pos_real - foot_pos_real)

                        if dist_real < closest_dist and dist_real < 3.0:
                            closest_dist = dist_real
                            closest_player = {
                                'track_id': int(track_id),
                                'bbox': (px1, py1, px2, py2),
                                'team': self.tracker_id_to_team.get(int(track_id), -1),
                                'foot_pos_real': foot_pos_real,
                            }

                    if closest_player is not None:
                        # MUDANÇA: passa speed_mag (velocidade) em vez de accel_mag
                        # MUDANÇA: passa pitch_mapper para goal_pos consistente
                        is_shot, shot_data = self.shot_detector.detect_shot(
                            ball_pos=ball_pos_real,
                            frame_time=frame_time,
                            speed_mag=speed_mag,
                            closest_player=closest_player,
                            pitch_mapper=self.pitch_mapper
                        )

                        if is_shot:
                            team_id = closest_player['team']

                            xg_value = None
                            if self.xg_model.model is not None:
                                # Coleta defensores do time adversário para def_pressure
                                defenders_pos = []
                                for xyxy2, cls2, tid2 in zip(
                                    sv_detections.xyxy,
                                    sv_detections.class_id,
                                    sv_detections.tracker_id
                                ):
                                    if tid2 is None or cls2 != 2:
                                        continue
                                    opp_team = self.tracker_id_to_team.get(int(tid2), -1)
                                    if opp_team != team_id and opp_team != -1:
                                        cx2 = int((xyxy2[0] + xyxy2[2]) / 2)
                                        cy2 = int((xyxy2[1] + xyxy2[3]) / 2)
                                        pos2 = self.pitch_mapper.frame_to_pitch(
                                            self.transformer, np.array([[cx2, cy2]])
                                        )[0]
                                        defenders_pos.append(pos2)

                                defenders_arr = np.array(defenders_pos) if defenders_pos else None
                                feat = self.xg_model.extract_features(
                                    np.array(ball_pos_real),
                                    body_part=1,
                                    defenders_positions=defenders_arr
                                )
                                xg_value = float(
                                    self.xg_model.predict_xg(feat.reshape(1, -1))[0]
                                )

                            shot_event = {
                                'time': frame_time,
                                'frame': self.frame_count,
                                'player_id': closest_player['track_id'],
                                'team': team_id,
                                'position': ball_pos_real.tolist(),
                                'velocity': float(speed_mag),
                                'xg': xg_value,
                            }
                            self.shot_events.append(shot_event)

                            if xg_value is not None:
                                print(f"\nShot at {frame_time:.1f}s! xG={xg_value:.3f} "
                                      f"conf={shot_data.get('confidence', 0):.2f}")
                            else:
                                print(f"\nShot at {frame_time:.1f}s!")

                            shot_tuple = (closest_player['track_id'], 0.0, "detected")
                            frame = draw_shot_event(frame, shot_tuple, xg_value)

                            # Radar overlay no canto inferior direito
                            pitch_xy = []
                            team_ids_arr = []
                            for xyxy2, cls2, tid2 in zip(
                                sv_detections.xyxy,
                                sv_detections.class_id,
                                sv_detections.tracker_id
                            ):
                                if tid2 is None:
                                    continue
                                cx2 = int((xyxy2[0] + xyxy2[2]) / 2)
                                cy2 = int((xyxy2[1] + xyxy2[3]) / 2)
                                pos2 = self.pitch_mapper.frame_to_pitch(
                                    self.transformer, np.array([[cx2, cy2]])
                                )[0]
                                pitch_xy.append(pos2)
                                team_ids_arr.append(
                                    self.tracker_id_to_team.get(int(tid2), 0)
                                )

                            if len(pitch_xy) >= 2:
                                pitch_xy = np.array(pitch_xy)
                                radar = create_radar_view(
                                    pitch_xy, np.array(team_ids_arr), ball_pos_real
                                )
                                rh, rw = radar.shape[:2]
                                new_w = int(rw * 0.3)
                                new_h = int(rh * 0.3)
                                radar_small = cv2.resize(radar, (new_w, new_h))
                                h_fr, w_fr = frame.shape[:2]
                                frame[h_fr-new_h:h_fr, w_fr-new_w:w_fr] = radar_small

                # Posição da bola no frame
                h_fr, w_fr = frame.shape[:2]
                pos_text = f"Ball: ({ball_pos_real[0]:.1f}, {ball_pos_real[1]:.1f})"
                cv2.putText(frame, pos_text, (10, h_fr - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, pos_text, (10, h_fr - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 6. Anotação visual das detecções
            if len(sv_detections) > 0 and sv_detections.tracker_id is not None:
                detections_dict = {
                    'balls': [], 'players': [], 'goalkeepers': [], 'referees': []
                }

                for xyxy, conf, cls_id, track_id in zip(
                    sv_detections.xyxy, sv_detections.confidence,
                    sv_detections.class_id, sv_detections.tracker_id
                ):
                    x1, y1, x2, y2 = map(int, xyxy)
                    track_id_int = int(track_id) if track_id is not None else -1
                    team_id = self.tracker_id_to_team.get(track_id_int, -1)

                    if cls_id == 0:
                        detections_dict['balls'].append(
                            (x1, y1, x2, y2, float(conf))
                        )
                    elif cls_id == 2:
                        detections_dict['players'].append(
                            (x1, y1, x2, y2, float(conf), int(cls_id), track_id_int, team_id)
                        )
                    elif cls_id == 1:
                        detections_dict['goalkeepers'].append(
                            (x1, y1, x2, y2, float(conf), int(cls_id), track_id_int, team_id)
                        )
                    elif cls_id == 3:
                        detections_dict['referees'].append(
                            (x1, y1, x2, y2, float(conf), int(cls_id), track_id_int)
                        )

                frame = draw_detections_on_frame(frame, detections_dict)

            writer.write(frame)
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        print(f"\nProcessing complete! Output: {output_path}")
        print(f"Detected {len(self.shot_events)} shot events")

        if self.shot_events:
            shot_map_path = output_path.replace('.mp4', '_shot_map.png')
            self._save_shot_map(shot_map_path)

        return self.shot_events

    def train_xg_model(self, statsbomb_path: str = None):
        """Train xG model — statsbomb_path ignorado, usa config.yaml."""
        print("\nTraining xG model using StatsBomb data from config...")
        metrics = self.xg_model.train()
        print(f"xG model trained! AUC={metrics['auc']:.4f}")

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
                # MUDANÇA: coleta só players e goalkeepers separadamente
                # (evita misturar árbitros no treino do classifier)
                for det in players:
                    x1, y1, x2, y2, conf, cls_id = det
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        crops.append(crop)
                for det in goalkeepers:
                    x1, y1, x2, y2, conf, cls_id = det
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        crops.append(crop)
            frame_idx += 1

        cap.release()
        print(f"Collected {len(crops)} crops")
        self.team_classifier.train(crops)

    def _save_shot_map(self, output_path: str):
        """Gera e salva imagem com todos os chutes marcados no campo."""
        from sports.configs.soccer import SoccerPitchConfiguration
        config = SoccerPitchConfiguration()

        import supervision as sv
        from sports.annotators.soccer import draw_pitch
        shot_map = draw_pitch(config, background_color=sv.Color.from_hex('4a7c3f'))

        for event in self.shot_events:
            pos = np.array(event['position'])
            xg = event['xg'] or 0.0
            shot_map = draw_pitch_with_xg(shot_map, pos, xg, config)

        cv2.imwrite(output_path, shot_map)
        print(f"Shot map saved to {output_path}")


if __name__ == "__main__":
    pipeline = VarzeaVisionPipeline()

    # xG já treinado — só carrega o .joblib salvo
    # Se quiser retreinar: apague o .joblib e descomente a linha abaixo
    # pipeline.train_xg_model()

    # Treina team classifier só se não existir
    if not Path("outputs/team_classifier.joblib").exists():
        pipeline.auto_train_team_classifier("data/LIVERPOOLxREAL2022teste.mp4")

    # Smoke test — 5 segundos
    events = pipeline.process_clip("data/test_15s.mp4")
    print(f"Shot events: {events}")