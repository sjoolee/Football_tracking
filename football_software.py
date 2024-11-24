import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd

def calculate_bbox_center(bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    return int((x_min + x_max) / 2), int((y_min + y_max) / 2)

def calculate_bbox_width(bounding_box):
    return bounding_box[2] - bounding_box[0]

def calculate_euclidean_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def calculate_directional_distance(point1, point2):
    return point1[0] - point2[0], point1[1] - point2[1]

def calculate_foot_position(bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    return int((x_min + x_max) / 2), int(y_max)

class BallPossessionTracker:
    def __init__(self):
        self.possession_threshold = 70
    
    def determine_ball_possession(self, player_locations, ball_location):
        ball_center = calculate_bbox_center(ball_location)
        min_distance = float('inf')
        possessing_player = -1

        for player_id, player_data in player_locations.items():
            player_bbox = player_data['bbox']
            left_foot_distance = calculate_euclidean_distance((player_bbox[0], player_bbox[-1]), ball_center)
            right_foot_distance = calculate_euclidean_distance((player_bbox[2], player_bbox[-1]), ball_center)
            closest_distance = min(left_foot_distance, right_foot_distance)

            if closest_distance < self.possession_threshold:
                if closest_distance < min_distance:
                    min_distance = closest_distance
                    possessing_player = player_id

        return possessing_player
    
class GameTracker:
    def __init__(self, model_path):
        self.object_detector = YOLO(model_path) 
        self.motion_tracker = sv.ByteTrack()

    def update_position_data(self, tracking_data):
        for entity_type, entity_frames in tracking_data.items():
            for frame_idx, frame_data in enumerate(entity_frames):
                for entity_id, entity_info in frame_data.items():
                    bbox = entity_info['bbox']
                    if entity_type == 'ball':
                        position = calculate_bbox_center(bbox)
                    else:
                        position = calculate_foot_position(bbox)
                    tracking_data[entity_type][frame_idx][entity_id]['position'] = position

    def interpolate_ball_trajectory(self, ball_positions):
        ball_coords = [pos.get(1, {}).get('bbox', []) for pos in ball_positions]
        ball_df = pd.DataFrame(ball_coords, columns=['x_min', 'y_min', 'x_max', 'y_max'])

        # Interpolate missing values
        ball_df = ball_df.interpolate()
        ball_df = ball_df.bfill()

        interpolated_positions = [{1: {"bbox": coords}} for coords in ball_df.to_numpy().tolist()]
        return interpolated_positions

    def process_frame_batch(self, video_frames):
        batch_size = 20 
        detection_results = [] 
        for i in range(0, len(video_frames), batch_size):
            batch_detections = self.object_detector.predict(video_frames[i:i+batch_size], conf=0.1)
            detection_results.extend(batch_detections)
        return detection_results

    def create_tracking_data(self, video_frames, use_cache=False, cache_path=None):
        if use_cache and cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                return pickle.load(cache_file)

        detection_results = self.process_frame_batch(video_frames)

        tracking_data = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_idx, frame_detection in enumerate(detection_results):
            class_labels = frame_detection.names
            class_indices = {label: idx for idx, label in class_labels.items()}

            # Convert to supervision format
            sv_detection = sv.Detections.from_ultralytics(frame_detection)

            # Convert goalkeeper to player class id
            for obj_idx, class_id in enumerate(sv_detection.class_id):
                if class_labels[class_id] == "goalkeeper":
                    sv_detection.class_id[obj_idx] = class_indices["player"]

            tracked_objects = self.motion_tracker.update_with_detections(sv_detection)

            tracking_data["players"].append({})
            tracking_data["referees"].append({})
            tracking_data["ball"].append({})

            for detected_object in tracked_objects:
                bbox = detected_object[0].tolist()
                class_id = detected_object[3]
                track_id = detected_object[4]

                if class_id == class_indices['player']:
                    tracking_data["players"][frame_idx][track_id] = {"bbox": bbox}
                elif class_id == class_indices['referee']:
                    tracking_data["referees"][frame_idx][track_id] = {"bbox": bbox}
            
            # ball detections
            for detected_object in sv_detection:
                bbox = detected_object[0].tolist()
                class_id = detected_object[3]

                if class_id == class_indices['ball']:
                    tracking_data["ball"][frame_idx][1] = {"bbox": bbox}

        if cache_path:
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(tracking_data, cache_file)

        return tracking_data
    
    def draw_player_indicator(self, frame, bounding_box, color, player_id=None):
        foot_y = int(bounding_box[3])
        center_x, _ = calculate_bbox_center(bounding_box)
        indicator_width = calculate_bbox_width(bounding_box)

        # Draw semi-circular indicator
        cv2.ellipse(
            frame,
            center=(center_x, foot_y),
            axes=(int(indicator_width), int(0.35 * indicator_width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw player ID if provided
        if player_id is not None:
            label_width = 40
            label_height = 20
            label_x1 = center_x - label_width // 2
            label_x2 = center_x + label_width // 2
            label_y1 = (foot_y - label_height // 2) + 15
            label_y2 = (foot_y + label_height // 2) + 15

            cv2.rectangle(frame, (int(label_x1), int(label_y1)),(int(label_x2), int(label_y2)), color, cv2.FILLED)
            text_x = label_x1 + (2 if player_id > 99 else 12)
            cv2.putText(frame,f"{player_id}",(int(text_x), int(label_y1 + 15)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0), 2)

        return frame

    def draw_ball_indicator(self, frame, bounding_box, color):
        top_y = int(bounding_box[1])
        center_x, _ = calculate_bbox_center(bounding_box)
        triangle_vertices = np.array([[center_x, top_y],[center_x - 10, top_y - 20], [center_x + 10, top_y - 20],])
        cv2.drawContours(frame, [triangle_vertices], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_vertices], 0, (0, 0, 0), 2)
        return frame

    def create_annotated_frames(self, video_frames, tracking_data):
        annotated_frames = []
        for frame_idx, frame in enumerate(video_frames):
            annotated_frame = frame.copy()

            player_data = tracking_data["players"][frame_idx]
            ball_data = tracking_data["ball"][frame_idx]
            referee_data = tracking_data["referees"][frame_idx]

            for player_id, player_info in player_data.items():
                team_color = player_info.get("team_color", (0, 0, 255))
                annotated_frame = self.draw_player_indicator(annotated_frame, player_info["bbox"], team_color, player_id)
                if player_info.get('has_ball', False):
                    annotated_frame = self.draw_ball_indicator(annotated_frame, player_info["bbox"],(0, 0, 255))

            for referee_info in referee_data.values():
                annotated_frame = self.draw_player_indicator(annotated_frame, referee_info["bbox"],(0, 0, 0))
            
            for ball_info in ball_data.values():
                annotated_frame = self.draw_ball_indicator(annotated_frame, ball_info["bbox"],(0, 255, 0))

            annotated_frames.append(annotated_frame)
        return annotated_frames

def load_video_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_list = []
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_list.append(frame)
    video_capture.release()
    return frame_list

def save_video_frames(frame_sequence, output_path):
    video_codec = cv2.VideoWriter_fourcc(*'XVID')
    frame_height, frame_width = frame_sequence[0].shape[:2]
    video_writer = cv2.VideoWriter(output_path, video_codec, 24, (frame_width, frame_height))
    for frame in frame_sequence:
        video_writer.write(frame)
    video_writer.release()

def main():
    video_frames = load_video_frames('input_videos/football.mp4')
    game_tracker = GameTracker('model/best.pt')
    tracking_data = game_tracker.create_tracking_data(
        video_frames, 
        use_cache=True, 
        cache_path='pickle/tracks_stubs.pkl'
    )
    tracking_data["ball"] = game_tracker.interpolate_ball_trajectory(tracking_data["ball"])
 
    possession_tracker = BallPossessionTracker()
    for frame_idx, player_positions in enumerate(tracking_data['players']):
        ball_location = tracking_data['ball'][frame_idx][1]['bbox']
        possessing_player = possession_tracker.determine_ball_possession(
            player_positions, 
            ball_location
        )
        if possessing_player != -1:
            tracking_data['players'][frame_idx][possessing_player]['has_ball'] = True

    annotated_frames = game_tracker.create_annotated_frames(video_frames, tracking_data)
    save_video_frames(annotated_frames, 'output_videos/soccer_output_video.avi')

if __name__ == '__main__':
    main()
