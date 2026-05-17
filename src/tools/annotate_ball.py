#!/usr/bin/env python3
"""
Script de Anotacao manual de posição da bola em vídeos de futebol.
Usuário desenha bbox ao redor da bola com mouse.
"""

import cv2
import csv
import os
from pathlib import Path


class BallAnnotator:
    def __init__(self, video_path: str, output_csv: str = None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_csv is None:
            video_name = Path(video_path).stem
            output_csv = f"src/data/annotations/{video_name}_ball_annotations.csv"
        self.output_csv = output_csv

        self.current_frame_idx = 0
        self.annotations = []
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.rect = None
        self.frame = None
        self.clone = None

    def load_annotations(self):
        if os.path.exists(self.output_csv):
            with open(self.output_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.annotations.append({
                        'frame': int(row['frame']),
                        'x1': float(row['x1']),
                        'y1': float(row['y1']),
                        'x2': float(row['x2']),
                        'y2': float(row['y2']),
                        'visible': int(row['visible'])
                    })
            print(f"Carregadas {len(self.annotations)} anotações existentes")
            if self.annotations:
                self.current_frame_idx = max(a['frame'] for a in self.annotations) + 1

    def save_annotations(self):
        Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, 'w', newline='') as f:
            fieldnames = ['frame', 'x1', 'y1', 'x2', 'y2', 'visible']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ann in self.annotations:
                writer.writerow(ann)
        print(f"Anotações salvas em: {self.output_csv}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.rect = (x, y, x, y)
            self.clone = self.frame.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.frame = self.clone.copy()
                cv2.rectangle(self.frame, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect = (self.start_x, self.start_y, x, y)

    def get_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame.copy()
            self.clone = frame.copy()
        return ret, frame

    def run(self):
        cv2.namedWindow("Anotacao de Bola")
        cv2.setMouseCallback("Anotacao de Bola", self.mouse_callback)

        self.load_annotations()

        print(f"\n{'='*60}")
        print("INSTRUÇÕES:")
        print("  Click + Drag  -> Desenhar bbox ao redor da bola")
        print("  ENTER         -> Confirmar Anotacao -> próximo frame")
        print("  SPACE         -> Pular frame (bola não visível)")
        print("  BACKSPACE     -> Voltar frame anterior")
        print("  Q             -> Sair e salvar")
        print(f"{'='*60}\n")

        while self.current_frame_idx < self.total_frames:
            ret, frame = self.get_frame(self.current_frame_idx)
            if not ret:
                break

            self.frame = frame.copy()
            self.clone = frame.copy()
            self.rect = None
            self.drawing = False

            # Carrega Anotacao existente, se houver
            ann = next((a for a in self.annotations if a['frame'] == self.current_frame_idx), None)
            if ann and ann['visible']:
                cv2.rectangle(self.frame,
                             (int(ann['x1']), int(ann['y1'])),
                             (int(ann['x2']), int(ann['y2'])),
                             (0, 255, 0), 2)
                # Atualiza o clone também para não perder o desenho antigo ao mover o mouse
                self.clone = self.frame.copy() 

            # Loop de renderização do frame atual (espera ação do usuário)
            while True:
                # Cria uma cópia temporária para exibir os textos sem "borrar" o desenho original
                display_frame = self.frame.copy()

                time_sec = self.current_frame_idx / self.fps
                info_text = f"Frame: {self.current_frame_idx}/{self.total_frames} | Time: {time_sec:.2f}s"
                cv2.putText(display_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                help_text = "ENTER:Confirmar | SPACE:Pular | Q:Sair"
                cv2.putText(display_frame, help_text, (10, self.height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Anotacao de Bola", display_frame)

                # Espera apenas 10 milissegundos. Se nenhuma tecla for pressionada, o loop continua e atualiza a tela
                key = cv2.waitKey(10) & 0xFF
                
                if key != 255:  # Se o usuário apertou alguma tecla, sai do loop interno para processar
                    break

            if key == ord('q') or key == ord('Q'):
                print("Saindo...")
                break

            elif key == 13: # ENTER
                if self.rect:
                    x1, y1, x2, y2 = self.rect
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    if x2 - x1 > 5 and y2 - y1 > 5:
                        ann = {'frame': self.current_frame_idx, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'visible': 1}
                        self.annotations = [a for a in self.annotations if a['frame'] != self.current_frame_idx]
                        self.annotations.append(ann)
                        self.current_frame_idx += 1
                    else:
                        print("Bbox muito pequena, ignore")
                else:
                    print("Nenhuma bbox desenhada")

            elif key == 32: # SPACE
                ann = {'frame': self.current_frame_idx, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'visible': 0}
                self.annotations = [a for a in self.annotations if a['frame'] != self.current_frame_idx]
                self.annotations.append(ann)
                self.current_frame_idx += 1
                print(f"Frame {self.current_frame_idx} marcado como não visível")

            elif key == 8: # BACKSPACE
                if self.current_frame_idx > 0:
                    self.current_frame_idx -= 1
                    print(f"Voltando para frame {self.current_frame_idx}")

        self.cap.release()
        cv2.destroyAllWindows()
        self.save_annotations()
        print(f"\nTotal de anotações: {len([a for a in self.annotations if a['visible']])}")
        print("Feito!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python annotate_ball.py <video_path> [output_csv]")
        print("Exemplo: python annotate_ball.py src/data/SPAINxCROATIA2.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    annotator = BallAnnotator(video_path, output_csv)
    annotator.run()