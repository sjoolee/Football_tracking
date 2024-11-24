import cv2
import numpy as np
import pandas as pd
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate the center point of the bounding box."""
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))
    
    @property
    def width(self) -> float:
        """Calculate width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def foot_position(self) -> Tuple[int, int]:
        """Calculate the position of feet (bottom center of bbox)."""
        return (int((self.x1 + self.x2) / 2), int(self.y2))

class GeometryUtils:
    @staticmethod
    def measure_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def measure_xy_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate separate x and y distances between two points."""
        return p1[0] - p2[0], p1[1] - p2[1]

class PlayerBallAssigner:
    def __init__(self, max_distance: float = 70):
        self.max_player_ball_distance = max_distance
    
    def assign_ball_to_player(self, players: Dict, ball_bbox: BoundingBox) -> int:
        """Assign ball to the nearest player within maximum distance."""
        ball_position = ball_bbox.center
        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = BoundingBox(*player['bbox'])
            distance_left = GeometryUtils.measure_distance(
                (player_bbox.x1, player_bbox.y2), 
                ball_position
            )
            distance_right = GeometryUtils.measure_distance(
                (player_bbox.x2, player_bbox.y2), 
                ball_position
            )
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player

class VideoProcessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def read_video(self) -> List[np.ndarray]:
        """Read video file and return list of frames."""
        cap = cv2.VideoCapture(str(self.input_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def save_video(self, frames: List[np.ndarray], fps: int = 24):
        """Save frames as video file."""
        if not frames:
            raise ValueError("No frames to save")
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        for frame in frames:
            out.write(frame)
        out.release()

class ObjectTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames: List[np.ndarray], batch_size: int = 20, conf: float = 0.1) -> List:
        """Detect objects in frames using YOLO model."""
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_detections = self.model.predict(
                frames[i:i + batch_size],
                conf=conf
            )
            detections.extend(batch_detections)
        return detections

    def process_tracks(self, frames: List[np.ndarray], stub_path: Optional[str] = None) -> Dict:
        """Process video frames and return tracked objects."""
        if stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for detection in detections:
            frame_tracks = self._process_single_frame(detection)
            for key in tracks:
                tracks[key].append(frame_tracks[key])

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def _process_single_frame(self, detection) -> Dict:
        """Process single frame detections."""
        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        detection_sv = sv.Detections.from_ultralytics(detection)
        
        # Convert goalkeeper to player
        detection_sv.class_id[detection_sv.class_id == cls_names_inv["goalkeeper"]] = cls_names_inv["player"]
        
        tracked_detections = self.tracker.update_with_detections(detection_sv)
        
        frame_tracks = {
            "players": {},
            "referees": {},
            "ball": {}
        }
        
        # Process tracked objects
        for det in tracked_detections:
            bbox = det[0].tolist()
            cls_id = det[3]
            track_id = det[4]
            
            if cls_id == cls_names_inv['player']:
                frame_tracks["players"][track_id] = {"bbox": bbox}
            elif cls_id == cls_names_inv['referee']:
                frame_tracks["referees"][track_id] = {"bbox": bbox}
                
        # Process ball (untracked)
        for det in detection_sv:
            if det[3] == cls_names_inv['ball']:
                frame_tracks["ball"][1] = {"bbox": det[0].tolist()}
                
        return frame_tracks

    def interpolate_ball_positions(self, ball_tracks: List[Dict]) -> List[Dict]:
        """Interpolate missing ball positions."""
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_tracks]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]

class Visualizer:
    @staticmethod
    def draw_ellipse(frame: np.ndarray, bbox: BoundingBox, color: Tuple[int, int, int], 
                     track_id: Optional[int] = None) -> np.ndarray:
        """Draw ellipse and optional ID for tracked objects."""
        x_center, y2 = bbox.center[0], int(bbox.y2)
        width = bbox.width

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rect_width, rect_height = 40, 20
            x1_rect = x_center - rect_width // 2
            y1_rect = y2 + 5
            
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x1_rect + rect_width), int(y1_rect + rect_height)),
                color,
                cv2.FILLED
            )
            
            text_x = x1_rect + (2 if track_id > 99 else 12)
            cv2.putText(
                frame,
                f"{track_id}",
                (int(text_x), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    @staticmethod
    def draw_triangle(frame: np.ndarray, bbox: BoundingBox, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw triangle indicator for ball possession."""
        x, y = bbox.center[0], int(bbox.y1)
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

def main():
    # Configuration
    INPUT_VIDEO = 'input_videos/08fd33_4.mp4'
    OUTPUT_VIDEO = 'output_videos/outputz.avi'
    MODEL_PATH = 'model/best.pt'
    STUB_PATH = 'pickle/track_stubs.pkl'

    # Initialize components
    video_processor = VideoProcessor(INPUT_VIDEO, OUTPUT_VIDEO)
    tracker = ObjectTracker(MODEL_PATH)
    visualizer = Visualizer()
    player_assigner = PlayerBallAssigner()

    # Process video
    frames = video_processor.read_video()
    tracks = tracker.process_tracks(frames, stub_path=STUB_PATH)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Process ball possession
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = BoundingBox(*tracks['ball'][frame_num][1]['bbox'])
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

    # Draw annotations and save video
    annotated_frames = []
    for frame_num, frame in enumerate(frames):
        frame = frame.copy()
        
        # Draw players
        for track_id, player in tracks["players"][frame_num].items():
            bbox = BoundingBox(*player["bbox"])
            color = player.get("team_color", (0, 0, 255))
            frame = visualizer.draw_ellipse(frame, bbox, color, track_id)
            
            if player.get('has_ball', False):
                frame = visualizer.draw_triangle(frame, bbox, (0, 0, 255))

        # Draw referees
        for _, referee in tracks["referees"][frame_num].items():
            bbox = BoundingBox(*referee["bbox"])
            frame = visualizer.draw_ellipse(frame, bbox, (0, 0, 0))

        # Draw ball
        for _, ball in tracks["ball"][frame_num].items():
            bbox = BoundingBox(*ball["bbox"])
            frame = visualizer.draw_triangle(frame, bbox, (0, 255, 0))

        annotated_frames.append(frame)

    video_processor.save_video(annotated_frames)

if __name__ == '__main__':
    main()