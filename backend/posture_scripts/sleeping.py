import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def determine_sleeping_position(landmarks):
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    eyes_mid_x = (left_eye.x + right_eye.x) / 2
    shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
    face_orient = abs(nose.x - eyes_mid_x)
    
    return "back" if face_orient < 0.05 and shoulder_y_diff < 0.1 else "side"

def get_sleeping_posture_feedback(landmarks, image_shape, calibration_data=None):
    h, w = image_shape[:2]
    visibility_threshold = 0.7
    key_points = {
        'nose': mp_pose.PoseLandmark.NOSE,
        'left_eye': mp_pose.PoseLandmark.LEFT_EYE,
        'right_eye': mp_pose.PoseLandmark.RIGHT_EYE,
        'left_ear': mp_pose.PoseLandmark.LEFT_EAR,
        'right_ear': mp_pose.PoseLandmark.RIGHT_EAR,
        'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
        'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
        'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
        'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
        'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
        'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
        'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
        'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
        'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
    }
    
    points = {}
    for key, idx in key_points.items():
        lm = landmarks[idx.value]
        if lm.visibility < visibility_threshold:
            return {'visibility_issue': True, 'missing_points': key}
        points[key] = np.array([lm.x * w, lm.y * h])
    
    shoulder_mid = (points['left_shoulder'] + points['right_shoulder']) / 2
    hip_mid = (points['left_hip'] + points['right_hip']) / 2
    points.update({'shoulder_mid': shoulder_mid, 'hip_mid': hip_mid})
    
    position = determine_sleeping_position(landmarks)
    head_point = points['nose'] if position == "back" else (points['left_ear'] if points['left_ear'][1] < points['right_ear'][1] else points['right_ear'])
    
    # Spine and Neck Alignment
    spine_angle = calculate_angle(head_point, shoulder_mid, hip_mid)
    
    # Head Tilt (for back sleeping)
    head_tilt = calculate_angle(points['left_eye'], points['nose'], points['right_eye']) if position == "back" else None
    
    # Shoulder Symmetry
    shoulder_slope = abs(np.arctan2(points['right_shoulder'][1] - points['left_shoulder'][1], 
                                   points['right_shoulder'][0] - points['left_shoulder'][0]) * 180 / np.pi)
    
    # Arm Positioning
    left_arm_angle = calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist'])
    right_arm_angle = calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist'])
    
    # Leg Positioning
    left_leg_angle = calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle'])
    right_leg_angle = calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle'])
    
    # Calibration Adjustments
    cal = calibration_data or {}
    spine_ref = cal.get('spine_angle', 180)
    head_ref = cal.get('head_tilt', 180) if position == "back" else None
    leg_ref = cal.get('leg_angle', 180 if position == "back" else 135)
    
    # Scoring Thresholds
    def score_value(value, excellent, good, fair, ref):
        delta = abs(value - ref)
        if delta <= excellent: return 100, "Excellent"
        elif delta <= good: return 80, "Good"
        elif delta <= fair: return 60, "Fair"
        else: return 30, "Poor"
    
    spine_score, spine_status = score_value(spine_angle, 5, 10, 15, spine_ref)
    spine_feedback = {
        "Excellent": "Spine perfectly aligned",
        "Good": "Spine alignment is good",
        "Fair": "Adjust head or hips slightly",
        "Poor": "Significant spine correction needed"
    }[spine_status]
    
    head_score = None
    head_feedback = ""
    if position == "back":
        head_score, head_status = score_value(head_tilt, 5, 10, 15, head_ref)
        head_feedback = {
            "Excellent": "Head well centered",
            "Good": "Head slightly tilted",
            "Fair": "Reduce head tilt",
            "Poor": "Head significantly tilted"
        }[head_status]
    
    shoulder_score, shoulder_status = score_value(shoulder_slope, 5, 10, 15, 0)
    shoulder_feedback = "Adjust shoulders" if shoulder_score < 70 else "Shoulders balanced"
    
    arm_score = (min(left_arm_angle, 180) + min(right_arm_angle, 180)) / 2
    arm_status = "Excellent" if arm_score > 150 else "Good" if arm_score > 120 else "Poor"
    arm_feedback = "Relax arms" if arm_score < 120 else "Arms well positioned"
    
    leg_bounds = (170, 160, 150) if position == "back" else (90, 80, 70)
    left_leg_score, left_leg_status = score_value(left_leg_angle, 5, 10, 15, leg_ref)
    right_leg_score, right_leg_status = score_value(right_leg_angle, 5, 10, 15, leg_ref)
    legs_score = (left_leg_score + right_leg_score) / 2
    legs_feedback = "Adjust leg bend" if legs_score < 70 else "Legs comfortably positioned"
    
    # Overall Score
    weights = {'spine': 0.4, 'legs': 0.3, 'shoulder': 0.15, 'arm': 0.15}
    if position == "back":
        weights.update({'spine': 0.35, 'head': 0.15})
        overall_score = sum([spine_score * 0.35, head_score * 0.15, legs_score * 0.3, shoulder_score * 0.15, arm_score * 0.15])
    else:
        overall_score = sum([spine_score * 0.4, legs_score * 0.3, shoulder_score * 0.15, arm_score * 0.15])
    
    overall_status = "Excellent" if overall_score >= 90 else "Good" if overall_score >= 70 else "Poor"
    overall_color = (0, 255, 0) if overall_score >= 70 else (0, 0, 255)
    
    return {
        'position': position,
        'spine': {'score': spine_score, 'angle': spine_angle, 'feedback': spine_feedback, 'color': (0, 255, 0) if spine_score >= 70 else (0, 0, 255)},
        'head': {'score': head_score, 'angle': head_tilt, 'feedback': head_feedback} if position == "back" else None,
        'shoulder': {'score': shoulder_score, 'feedback': shoulder_feedback},
        'arms': {'score': arm_score, 'feedback': arm_feedback},
        'legs': {'score': legs_score, 'feedback': legs_feedback, 'angles': (left_leg_angle, right_leg_angle)},
        'overall': {'score': overall_score, 'status': overall_status, 'color': overall_color},
        'points': points
    }

def draw_visualization(image, feedback):
    points = feedback['points']
    pos = feedback['position']
    head_point = points['nose'] if pos == "back" else (points['left_ear'] if points['left_ear'][1] < points['right_ear'][1] else points['right_ear'])
    
    # Spine
    cv2.line(image, tuple(head_point.astype(int)), tuple(points['shoulder_mid'].astype(int)), feedback['spine']['color'], 3)
    cv2.line(image, tuple(points['shoulder_mid'].astype(int)), tuple(points['hip_mid'].astype(int)), feedback['spine']['color'], 3)
    cv2.putText(image, f"{feedback['spine']['angle']:.1f}°", tuple((points['shoulder_mid'] + 10).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['spine']['color'], 1)
    
    # Legs
    for side in ['left', 'right']:
        cv2.line(image, tuple(points[f'{side}_hip'].astype(int)), tuple(points[f'{side}_knee'].astype(int)), (255, 255, 255), 2)
        cv2.line(image, tuple(points[f'{side}_knee'].astype(int)), tuple(points[f'{side}_ankle'].astype(int)), (255, 255, 255), 2)
    
    # Arms
    for side in ['left', 'right']:
        cv2.line(image, tuple(points[f'{side}_shoulder'].astype(int)), tuple(points[f'{side}_elbow'].astype(int)), (200, 200, 200), 1)
        cv2.line(image, tuple(points[f'{side}_elbow'].astype(int)), tuple(points[f'{side}_wrist'].astype(int)), (200, 200, 200), 1)
    
    return image

def create_feedback_panel(image, feedback):
    h, w = image.shape[:2]
    panel_h = 120
    expanded = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    expanded[:h, :] = image
    expanded[h:, :] = (18, 18, 24)
    
    y_pos = h + 30
    cv2.putText(expanded, f"Position: {feedback['position'].capitalize()}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    cv2.putText(expanded, f"Spine: {feedback['spine']['score']} ({feedback['spine']['angle']:.1f}°)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback['spine']['color'], 2)
    if feedback['head']:
        y_pos += 30
        cv2.putText(expanded, f"Head: {feedback['head']['score']} ({feedback['head']['angle']:.1f}°)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if feedback['head']['score'] >= 70 else (0, 0, 255), 2)
    y_pos += 30
    cv2.putText(expanded, f"Legs: {feedback['legs']['score']}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(expanded, f"Overall: {feedback['overall']['score']}", (w - 150, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback['overall']['color'], 2)
    
    banner_h = 35
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay[h - banner_h:], 0.75, image[h - banner_h:], 0.25, 0, expanded[h - banner_h:])
    text = feedback.get('current_feedback', "Analyzing...")
    text_x = (w - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]) // 2
    cv2.putText(expanded, text, (text_x, h - banner_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback.get('feedback_color', (255, 255, 255)), 2)
    
    return expanded

class PostureCalibrationData:
    def __init__(self):
        self.calibrated = False
        self.data = {}
    
    def calibrate(self, feedback):
        self.data = {
            'spine_angle': feedback['spine']['angle'],
            'head_tilt': feedback['head']['angle'] if feedback['head'] else None,
            'leg_angle': sum(feedback['legs']['angles']) / 2
        }
        self.calibrated = True
    
    def get_data(self):
        return self.data if self.calibrated else None

class SleepingPostureCoach:
    def __init__(self):
        self.calibration = PostureCalibrationData()
        self.feedback_history = deque(maxlen=60)
        self.current_feedback = "Analyzing..."
        self.feedback_color = (255, 255, 255)
        self.last_feedback_time = 0
        self.feedback_interval = 3
    
    def update_feedback(self, feedback):
        scores = {k: v['score'] for k, v in feedback.items() if k in ['spine', 'head', 'shoulder', 'arms', 'legs'] and v}
        worst = min(scores, key=scores.get)
        self.feedback_history.append((worst, scores[worst]))
        
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_interval:
            counts = {k: sum(1 for x, s in self.feedback_history if x == k and s < 70) for k in scores}
            max_issue = max(counts, key=counts.get, default=None)
            if max_issue and counts[max_issue] > 10:
                self.current_feedback = feedback[max_issue]['feedback']
                self.feedback_color = feedback[max_issue].get('color', (255, 255, 255))
            elif feedback['overall']['score'] >= 90:
                self.current_feedback = "Excellent posture!"
                self.feedback_color = (0, 255, 0)
            self.last_feedback_time = current_time
    
    def run(self, camera_index=0, width=1280, height=720):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8, model_complexity=2) as pose:
            calibrating = False
            countdown = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                rgb.flags.writeable = True
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    feedback = get_sleeping_posture_feedback(landmarks, image.shape, self.calibration.get_data())
                    
                    if 'visibility_issue' in feedback:
                        cv2.putText(image, f"Missing: {feedback['missing_points']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        if calibrating:
                            if countdown > 0:
                                cv2.putText(image, f"Calibrating in {countdown}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                countdown -= 1
                            else:
                                self.calibration.calibrate(feedback)
                                calibrating = False
                                print("Calibration completed!")
                        else:
                            image = draw_visualization(image, feedback)
                            self.update_feedback(feedback)
                            feedback['current_feedback'] = self.current_feedback
                            feedback['feedback_color'] = self.feedback_color
                            image = create_feedback_panel(image, feedback)
                else:
                    cv2.putText(image, "No pose detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(image, "Press 'c' to calibrate, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow('SleepingPostureCoach', image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    calibrating = True
                    countdown = 3
                    print("Calibrating... Assume optimal posture.")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    coach = SleepingPostureCoach()
    coach.run()
