import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from modules.common import load_config, get_project_root
load_dotenv()
from modules.detection import FootballDetector
from modules.tracking import FootballTracker, BallKalmanTracker
from modules.field_detection import FieldDetector
from modules.homography import PitchMapper
from modules.shot_detection import ShotDetector
from modules.xg_model import XGModel
from modules.team_classifier import TeamClassifierWrapper
from modules.utils import (
    draw_detections_on_frame,
    draw_shot_event,
    create_radar_view,
    draw_pitch_with_xg,
)


class VarzeaVisionPipeline:
    def __init__(self, config_path: str = None):
        print("=" * 60)
        print("Initializing VarzeaVision Pipeline...")
        print("=" * 60)

        self.config = load_config(config_path)

        self.detector        = FootballDetector(config_path)
        self.tracker         = FootballTracker(config_path)
        # Opção 4: BallKalmanTracker substitui BallTracker simples.
        # Kalman Filter [x, y, vx, vy] rejeita detecções improváveis via
        # gate de Mahalanobis — elimina saltos para marcações estáticas do campo.
        self.ball_tracker    = BallKalmanTracker(
            max_missing_frames=8,
            process_noise=10.0,
            measurement_noise=25.0,
            gate_sigma=3.0,
        )
        self.field_detector  = FieldDetector(config_path)
        self.pitch_mapper    = PitchMapper(config_path)
        self.shot_detector   = ShotDetector(config_path)
        self.xg_model        = XGModel(config_path)
        self.team_classifier = TeamClassifierWrapper(config_path)

        self._last_H_nbjw     = None
        self._last_kp_quality = 0.0

        self.current_clip       = None
        self.transformer        = None
        self.frame_count        = 0
        self.shot_events        = []
        self.tracker_id_to_team = {}
        self.ball_positions     = {}

        print("Pipeline initialized successfully!\n")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _create_sv_detections_players(self, players, goalkeepers, referees) -> sv.Detections:
        arrays_to_stack = []
        if len(players) > 0:     arrays_to_stack.append(players)
        if len(goalkeepers) > 0: arrays_to_stack.append(goalkeepers)
        if len(referees) > 0:    arrays_to_stack.append(referees)

        if not arrays_to_stack:
            return sv.Detections.empty()

        all_data = np.vstack(arrays_to_stack)
        return sv.Detections(
            xyxy=all_data[:, :4],
            confidence=all_data[:, 4],
            class_id=all_data[:, 5].astype(int),
        )

    def _get_player_crops(self, frame, detections):
        crops, track_ids = [], []
        for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
            if track_id is None:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                track_ids.append(int(track_id))
        return crops, track_ids

    def _run_team_classification(self, frame, sv_detections):
        if sv_detections is None or len(sv_detections) == 0:
            return
        if sv_detections.tracker_id is None:
            return
        player_crops, track_ids = self._get_player_crops(frame, sv_detections)
        if not player_crops:
            return
        predictions = self.team_classifier.predict(player_crops)
        for track_id, pred in zip(track_ids, predictions):
            self.tracker_id_to_team[track_id] = pred

    def _resolve_goalkeepers(self, sv_detections, pitch_xy_real):
        if pitch_xy_real is None or sv_detections.tracker_id is None:
            return

        gk_mask     = sv_detections.class_id == 1
        player_mask = sv_detections.class_id == 2

        if not np.any(gk_mask) or not np.any(player_mask):
            return

        players_xy       = pitch_xy_real[player_mask]
        players_tids     = sv_detections.tracker_id[player_mask]
        gk_xy            = pitch_xy_real[gk_mask]
        gk_tids          = sv_detections.tracker_id[gk_mask]

        players_team_ids = np.array([
            self.tracker_id_to_team.get(int(tid), -1) for tid in players_tids
        ])

        if not (np.any(players_team_ids == 0) and np.any(players_team_ids == 1)):
            return

        gk_team_ids = self.team_classifier.resolve_goalkeepers(
            players_xy=players_xy,
            players_team_ids=players_team_ids,
            goalkeepers_xy=gk_xy,
        )

        for tid, team_id in zip(gk_tids, gk_team_ids):
            self.tracker_id_to_team[int(tid)] = int(team_id)

    # ── pipeline principal ───────────────────────────────────────────────────

    def process_clip(self, video_path: str, output_path: str = None):
        print(f"\n{'=' * 60}")
        print(f"Processing clip: {video_path}")
        print(f"{'=' * 60}\n")

        self.current_clip       = video_path
        self.shot_events        = []
        self.frame_count        = 0
        self.tracker_id_to_team = {}
        self.ball_positions     = {}

        self.tracker.reset()
        self.ball_tracker.reset()
        self.pitch_mapper.reset_clip()
        self.shot_detector.reset_clip()
        self.field_detector.reset_clip()
        self._last_H_nbjw     = None
        self._last_kp_quality = 0.0
        self.transformer      = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            output_path = str(get_project_root() / "outputs" / f"processed_{Path(video_path).stem}.mp4")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        pbar   = tqdm(total=total_frames, desc="Processing frames")

        FIELD_DETECTION_STRIDE = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame_time = self.frame_count / fps

            # 1. Detection
            balls, players, goalkeepers, referees = self.detector.get_detections(frame)

            # 2a. Tracking de jogadores
            sv_detections = self._create_sv_detections_players(players, goalkeepers, referees)
            if len(sv_detections) > 0:
                sv_detections = self.tracker.update(frame, sv_detections)

            # 2b. Tracking da bola com Kalman Filter
            best_ball = self.ball_tracker.update(balls)

            # 3. Team classification
            if self.frame_count == 1 or self.frame_count % 30 == 0:
                self._run_team_classification(frame, sv_detections)

            # 4. Homography
            if self.frame_count % FIELD_DETECTION_STRIDE == 1 or self._last_H_nbjw is None:
                H_nbjw, kp_quality    = self.field_detector.get_homography(frame)
                self._last_H_nbjw     = H_nbjw
                self._last_kp_quality = kp_quality
            else:
                H_nbjw    = self._last_H_nbjw
                kp_quality = self._last_kp_quality

            H = self.pitch_mapper.update(H_nbjw, kp_quality)
            self.transformer = H

            # Projeção global
            valid_tracking = (
                len(sv_detections) > 0 and sv_detections.tracker_id is not None
            )
            pitch_xy_real = None

            if self.transformer is not None and valid_tracking:
                centers_x   = (sv_detections.xyxy[:, 0] + sv_detections.xyxy[:, 2]) / 2
                feet_y      = sv_detections.xyxy[:, 3]
                feet_pixels = np.column_stack((centers_x, feet_y))
                pitch_xy_real = self.pitch_mapper.frame_to_pitch(self.transformer, feet_pixels)
                self._resolve_goalkeepers(sv_detections, pitch_xy_real)

            # 5. Shot detection
            if best_ball is not None and self.transformer is not None and self.pitch_mapper.is_warmed_up:
                ball_center   = self.detector.get_ball_center(best_ball)
                ball_pos_real = self.pitch_mapper.frame_to_pitch(
                    self.transformer, np.array([ball_center])
                )[0]

                self.ball_positions[frame_time] = ball_pos_real
                vel_vec, speed_mag = self.shot_detector.compute_ball_velocity(
                    ball_pos_real, frame_time
                )

                if valid_tracking and pitch_xy_real is not None:
                    players_mask = sv_detections.class_id == 2

                    if np.any(players_mask):
                        players_pos_real  = pitch_xy_real[players_mask]
                        players_track_ids = sv_detections.tracker_id[players_mask]
                        players_xyxy      = sv_detections.xyxy[players_mask]

                        dists        = np.linalg.norm(players_pos_real - ball_pos_real, axis=1)
                        min_idx      = np.argmin(dists)
                        closest_dist = dists[min_idx]

                        if closest_dist < self.shot_detector.max_player_ball_dist:
                            closest_tid = int(players_track_ids[min_idx])
                            team_id     = self.tracker_id_to_team.get(closest_tid, -1)

                            closest_player = {
                                'track_id':      closest_tid,
                                'bbox':          tuple(players_xyxy[min_idx].astype(int)),
                                'team':          team_id,
                                'foot_pos_real': players_pos_real[min_idx],
                            }

                            is_shot, shot_data = self.shot_detector.detect_shot(
                                ball_pos=ball_pos_real,
                                frame_time=frame_time,
                                speed_mag=speed_mag,
                                closest_player=closest_player,
                                pitch_mapper=self.pitch_mapper,
                            )

                            if is_shot:
                                xg_value = None
                                if self.xg_model.model is not None:
                                    opp_mask = np.array([
                                        self.tracker_id_to_team.get(int(tid), -1) != team_id
                                        and self.tracker_id_to_team.get(int(tid), -1) != -1
                                        for tid in sv_detections.tracker_id
                                    ])
                                    defenders_arr = (
                                        pitch_xy_real[opp_mask] if np.any(opp_mask) else None
                                    )
                                    feat = self.xg_model.extract_features(
                                        np.array(ball_pos_real),
                                        body_part=1,
                                        defenders_positions=defenders_arr,
                                    )
                                    xg_value = float(
                                        self.xg_model.predict_xg(feat.reshape(1, -1))[0]
                                    )

                                self.shot_events.append({
                                    'time':      frame_time,
                                    'frame':     self.frame_count,
                                    'player_id': closest_tid,
                                    'team':      team_id,
                                    'position':  ball_pos_real.tolist(),
                                    'velocity':  float(speed_mag),
                                    'xg':        xg_value,
                                    'body_part': 'foot',
                                })

                                shot_tuple = (closest_tid, float(shot_data['confidence']), "detected")
                                frame = draw_shot_event(frame, shot_tuple, xg_value)

                    # Radar overlay
                    if pitch_xy_real is not None and len(pitch_xy_real) >= 2:
                        team_ids_arr = np.array([
                            self.tracker_id_to_team.get(int(t), 0)
                            for t in sv_detections.tracker_id
                        ])
                        radar = create_radar_view(pitch_xy_real, team_ids_arr, ball_pos_real)
                        rh, rw      = radar.shape[:2]
                        new_w       = int(rw * 0.3)
                        new_h       = int(rh * 0.3)
                        radar_small = cv2.resize(radar, (new_w, new_h))
                        h_fr, w_fr  = frame.shape[:2]
                        if new_h <= h_fr and new_w <= w_fr:
                            frame[h_fr - new_h:h_fr, w_fr - new_w:w_fr] = radar_small

                h_fr, w_fr = frame.shape[:2]
                pos_text = f"Ball: ({ball_pos_real[0]:.1f}, {ball_pos_real[1]:.1f})"
                cv2.putText(frame, pos_text, (10, h_fr - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, pos_text, (10, h_fr - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 6. Anotação visual
            if valid_tracking:
                balls_for_draw = []
                if best_ball is not None:
                    balls_for_draw.append(tuple(best_ball[:5]))

                detections_dict = {
                    'balls': balls_for_draw,
                    'players': [],
                    'goalkeepers': [],
                    'referees': [],
                }
                for xyxy, conf, cls_id, track_id in zip(
                    sv_detections.xyxy,
                    sv_detections.confidence,
                    sv_detections.class_id,
                    sv_detections.tracker_id,
                ):
                    t_id    = int(track_id)
                    team_id = self.tracker_id_to_team.get(t_id, -1)
                    box     = (*map(int, xyxy), float(conf))

                    if   cls_id == 2: detections_dict['players'].append((*box, int(cls_id), t_id, team_id))
                    elif cls_id == 1: detections_dict['goalkeepers'].append((*box, int(cls_id), t_id, team_id))
                    elif cls_id == 3: detections_dict['referees'].append((*box, int(cls_id), t_id))

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

    # ── treino ───────────────────────────────────────────────────────────────

    def train_xg_model(self):
        print("\nTraining xG model using StatsBomb data from config...")
        metrics = self.xg_model.train()
        print(f"xG model trained! AUC={metrics['auc']:.4f}")

    def train_team_classifier(self, player_crops_dir: str):
        print(f"\nTraining team classifier using crops from: {player_crops_dir}")
        self.team_classifier.train(player_crops_dir)
        print("Team classifier trained and saved!")

    def auto_train_team_classifier(self, video_path: str, stride: int = 60):
        print("Auto-training team classifier from video...")
        cap       = cv2.VideoCapture(video_path)
        crops     = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                _, players, goalkeepers, _ = self.detector.get_detections(frame)
                src = []
                if len(players) > 0:     src.append(players)
                if len(goalkeepers) > 0: src.append(goalkeepers)
                if src:
                    for det in np.vstack(src):
                        x1, y1, x2, y2 = map(int, det[:4])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops.append(crop)
            frame_idx += 1

        cap.release()
        print(f"Collected {len(crops)} crops")
        self.team_classifier.train(crops)

    def _save_shot_map(self, output_path: str):
        from sports.configs.soccer import SoccerPitchConfiguration
        from sports.annotators.soccer import draw_pitch

        config   = SoccerPitchConfiguration()
        shot_map = draw_pitch(config, background_color=sv.Color.from_hex('4a7c3f'))

        for event in self.shot_events:
            pos = np.array(event['position'])
            xg  = event['xg'] or 0.0
            shot_map = draw_pitch_with_xg(shot_map, pos, xg, config)

        cv2.imwrite(output_path, shot_map)
        print(f"Shot map saved to {output_path}")


if __name__ == "__main__":
    pipeline = VarzeaVisionPipeline()

    print("\n" + "=" * 60)
    print("[1/2] TREINAMENTO DE RECONHECIMENTO DE UNIFORMES")
    print("=" * 60)
    pipeline.auto_train_team_classifier("src/data/SPAINxCROATIA1.mp4")

    print("\n" + "=" * 60)
    print("[2/2] INICIANDO PROCESSAMENTO TÁTICO")
    print("=" * 60)
    events = pipeline.process_clip("src/data/SPAINxCROATIA2.mp4")

    print("\nResumo Final de Eventos:")
    for ev in events:
        xg_str = f"xG: {ev['xg']:.3f}" if ev['xg'] is not None else "xG: N/A"
        print(f"Tempo: {ev['time']:.1f}s | Equipe: {ev['team']} | {xg_str}")