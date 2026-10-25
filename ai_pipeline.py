
import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_pipeline(video_file):
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=30, n_init=3)
    cap = cv2.VideoCapture(video_file)
    track_positions = {}

    for frame_num in range(100):
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, classes=[0], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed(): continue
            try: track_id = int(track.track_id)
            except: continue 
            ltrb = track.to_ltrb()
            cx = (ltrb[0] + ltrb[2]) / 2
            cy = (ltrb[1] + ltrb[3]) / 2
            if track_id not in track_positions: track_positions[track_id] = []
            track_positions[track_id].append((cx, cy))
    cap.release()
    return track_positions

if __name__ == "__main__":
    print("AI Pipeline Module Ready.")
