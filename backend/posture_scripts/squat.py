import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points with high precision."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def calculate_spine_deviation(hip, shoulder, ear):
    """Calculate deviation from straight spine alignment."""
    spine_points = [hip, shoulder, ear]
    deviation = 0
    for i in range(1, len(spine_points) - 1):
        angle = calculate_angle(spine_points[i-1], spine_points[i], spine_points[i+1])
        deviation += abs(180 - angle)
    return deviation

def get_posture_feedback(landmarks, image_shape, calibration_data=None):
    """Comprehensive squat posture analysis with scoring and feedback."""
    h, w = image_shape[:2]
    feedback = {}
    visibility_threshold = 0.7

    # Define key landmarks
    key_points = {
        'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        'left_hip': mp_pose.PoseLandmark.LEFT_HIP.value,
        'right_hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
        'left_knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
        'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,
        'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
        'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        'left_ear': mp_pose.PoseLandmark.LEFT_EAR.value,
        'right_ear': mp_pose.PoseLandmark.RIGHT_EAR.value,
        'left_foot_index': mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
        'right_foot_index': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
    }

    points = {}
    for key, idx in key_points.items():
        landmark = landmarks[idx]
        if landmark.visibility < visibility_threshold:
            return {'visibility_issue': True, 'missing_points': key}
        points[key] = np.array([landmark.x * w, landmark.y * h])

    # Calculate midpoints
    hip_mid = (points['left_hip'] + points['right_hip']) / 2
    shoulder_mid = (points['left_shoulder'] + points['right_shoulder']) / 2
    ear_mid = (points['left_ear'] + points['right_ear']) / 2
    points['hip_mid'] = hip_mid
    points['shoulder_mid'] = shoulder_mid
    points['ear_mid'] = ear_mid

    calibration_factor = calibration_data.get('calibration_factor', 1.0) if calibration_data else 1.0

    # 1. Knee Angle Analysis
    left_knee_angle = calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle'])
    right_knee_angle = calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle'])
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

    knee_score = 100
    if 85 <= avg_knee_angle <= 95:
        knee_status = "Excellent"
        knee_feedback = "Perfect knee angle!"
        knee_color = (0, 255, 0)
    elif 80 <= avg_knee_angle <= 100:
        knee_status = "Good"
        knee_feedback = "Good knee angle, aim for 90°"
        knee_color = (0, 255, 0)
        knee_score = 80
    elif 70 <= avg_knee_angle <= 110:
        knee_status = "Fair"
        knee_feedback = "Adjust knee angle closer to 90°"
        knee_color = (0, 165, 255)
        knee_score = 60
    else:
        knee_status = "Poor"
        knee_feedback = "Knee angle too far from 90°, adjust form"
        knee_color = (0, 0, 255)
        knee_score = 30

    # 2. Hip Depth Analysis
    hip_depth_score = 100
    left_hip_depth = (points['left_hip'][1] - points['left_knee'][1]) / h
    right_hip_depth = (points['right_hip'][1] - points['right_knee'][1]) / h
    avg_hip_depth = (left_hip_depth + right_hip_depth) / 2

    if avg_hip_depth > 0.05:
        hip_status = "Excellent"
        hip_feedback = "Great depth, hips below knees!"
        hip_color = (0, 255, 0)
    elif avg_hip_depth > 0:
        hip_status = "Good"
        hip_feedback = "Almost there, lower hips slightly"
        hip_color = (0, 255, 0)
        hip_depth_score = 80
    elif avg_hip_depth > -0.05:
        hip_status = "Fair"
        hip_feedback = "Lower hips below knee level"
        hip_color = (0, 165, 255)
        hip_depth_score = 60
    else:
        hip_status = "Poor"
        hip_feedback = "Squat deeper, hips must go below knees"
        hip_color = (0, 0, 255)
        hip_depth_score = 30

    # 3. Back Alignment Analysis
    left_spine_dev = calculate_spine_deviation(points['left_hip'], points['left_shoulder'], points['left_ear'])
    right_spine_dev = calculate_spine_deviation(points['right_hip'], points['right_shoulder'], points['right_ear'])
    avg_spine_dev = (left_spine_dev + right_spine_dev) / 2

    back_score = 100
    if avg_spine_dev < 10 * calibration_factor:
        back_status = "Excellent"
        back_feedback = "Spine perfectly straight!"
        back_color = (0, 255, 0)
    elif avg_spine_dev < 20 * calibration_factor:
        back_status = "Good"
        back_feedback = "Good spine alignment"
        back_color = (0, 255, 0)
        back_score = 80
    elif avg_spine_dev < 30 * calibration_factor:
        back_status = "Fair"
        back_feedback = "Keep spine straighter"
        back_color = (0, 165, 255)
        back_score = 60
    else:
        back_status = "Poor"
        back_feedback = "Straighten your spine, avoid rounding"
        back_color = (0, 0, 255)
        back_score = 30

    # 4. Knee Position Analysis
    knee_toe_threshold = 30 * calibration_factor  # pixels
    left_knee_over_toes = (points['left_knee'][0] - points['left_foot_index'][0]) > knee_toe_threshold
    right_knee_over_toes = (points['right_foot_index'][0] - points['right_knee'][0]) > knee_toe_threshold

    knee_pos_score = 100
    if not (left_knee_over_toes or right_knee_over_toes):
        knee_pos_status = "Excellent"
        knee_pos_feedback = "Knees perfectly aligned with toes!"
        knee_pos_color = (0, 255, 0)
    else:
        knee_pos_status = "Poor"
        knee_pos_feedback = "Keep knees behind toes"
        knee_pos_color = (0, 0, 255)
        knee_pos_score = 30

    # Overall Score
    weights = {'knee': 0.3, 'hip': 0.3, 'back': 0.3, 'knee_pos': 0.1}
    overall_score = int(
        weights['knee'] * knee_score +
        weights['hip'] * hip_depth_score +
        weights['back'] * back_score +
        weights['knee_pos'] * knee_pos_score
    )

    if overall_score >= 90:
        overall_status = "Excellent"
        overall_color = (0, 255, 0)
    elif overall_score >= 70:
        overall_status = "Good"
        overall_color = (0, 255, 0)
    elif overall_score >= 50:
        overall_status = "Fair"
        overall_color = (0, 165, 255)
    else:
        overall_status = "Poor"
        overall_color = (0, 0, 255)

    detailed_metrics = {
        'avg_knee_angle': avg_knee_angle,
        'avg_hip_depth': avg_hip_depth,
        'avg_spine_dev': avg_spine_dev,
        'left_knee_over_toes': left_knee_over_toes,
        'right_knee_over_toes': right_knee_over_toes
    }

    return {
        'knee': {'status': knee_status, 'feedback': knee_feedback, 'color': knee_color, 'score': knee_score},
        'hip': {'status': hip_status, 'feedback': hip_feedback, 'color': hip_color, 'score': hip_depth_score},
        'back': {'status': back_status, 'feedback': back_feedback, 'color': back_color, 'score': back_score},
        'knee_pos': {'status': knee_pos_status, 'feedback': knee_pos_feedback, 'color': knee_pos_color, 'score': knee_pos_score},
        'overall': {'status': overall_status, 'score': overall_score, 'color': overall_color},
        'points': points,
        'detailed_metrics': detailed_metrics
    }

def draw_posture_visualization(image, feedback):
    """Draw detailed squat posture guides with correction indicators."""
    points = feedback['points']
    h, w = image.shape[:2]

    # Draw spine line
    cv2.line(image, tuple(points['hip_mid'].astype(int)), tuple(points['shoulder_mid'].astype(int)), feedback['back']['color'], 3)
    cv2.line(image, tuple(points['shoulder_mid'].astype(int)), tuple(points['ear_mid'].astype(int)), feedback['back']['color'], 3)

    # Draw leg lines with knee angle color and over-toes indicator
    for side in ['left', 'right']:
        # Use knee angle color for hip-knee and knee-ankle lines
        cv2.line(image, tuple(points[f'{side}_hip'].astype(int)), tuple(points[f'{side}_knee'].astype(int)), feedback['knee']['color'], 3)
        cv2.line(image, tuple(points[f'{side}_knee'].astype(int)), tuple(points[f'{side}_ankle'].astype(int)), feedback['knee']['color'], 3)
        # Draw knee to foot index line in white
        cv2.line(image, tuple(points[f'{side}_knee'].astype(int)), tuple(points[f'{side}_foot_index'].astype(int)), (255, 255, 255), 1)
        # If knee is over toes, draw a red circle around the knee
        if feedback['detailed_metrics'][f'{side}_knee_over_toes']:
            cv2.circle(image, tuple(points[f'{side}_knee'].astype(int)), 10, (0, 0, 255), 2)

    # Draw hip depth indicator
    hip_y = int(min(points['left_hip'][1], points['right_hip'][1]))
    knee_y = int(max(points['left_knee'][1], points['right_knee'][1]))
    cv2.line(image, (w - 50, hip_y), (w - 30, hip_y), feedback['hip']['color'], 2)
    cv2.line(image, (w - 50, knee_y), (w - 30, knee_y), (255, 255, 255), 2)

    # Draw key points
    for key in ['hip_mid', 'shoulder_mid', 'ear_mid', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']:
        cv2.circle(image, tuple(points[key].astype(int)), 4, (255, 255, 0), -1)

    # Angle annotations
    metrics = feedback['detailed_metrics']
    cv2.putText(image, f"Knee: {metrics['avg_knee_angle']:.1f}°", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['knee']['color'], 1)
    cv2.putText(image, f"Spine Dev: {metrics['avg_spine_dev']:.1f}°", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['back']['color'], 1)

    return image

def create_feedback_panel(image, feedback, state, rep_count):
    """Create a premium feedback panel with scores and correction guidance."""
    h, w = image.shape[:2]
    panel_height = 130
    expanded_image = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    expanded_image[:h, :w] = image
    expanded_image[h:, :] = (18, 18, 24)

    # Gradient
    for y in range(h, h + 20):
        alpha = (y - h) / 20
        cv2.line(expanded_image, (0, y), (w, y), (20 + int(alpha * 10), 20 + int(alpha * 10), 26 + int(alpha * 10)), 1)

    # Score and rep count
    cv2.putText(expanded_image, f"Score: {feedback['overall']['score']}", (20, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback['overall']['color'], 2)
    cv2.putText(expanded_image, f"Reps: {rep_count} | State: {state}", (20, h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Progress bars
    bar_y = h + 80
    bar_max_width = w - 240
    def draw_bar(y, label, score, color, status):
        cv2.putText(expanded_image, label, (20, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(expanded_image, (90, y), (90 + bar_max_width, y + 18), (35, 35, 45), -1)
        bar_width = int((score / 100) * bar_max_width)
        cv2.rectangle(expanded_image, (90, y), (90 + bar_width, y + 18), color, -1)
        cv2.putText(expanded_image, status, (100 + bar_max_width, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    draw_bar(bar_y, "Knee:", feedback['knee']['score'], feedback['knee']['color'], feedback['knee']['status'])
    draw_bar(bar_y + 25, "Hip Depth:", feedback['hip']['score'], feedback['hip']['color'], feedback['hip']['status'])
    draw_bar(bar_y + 50, "Back:", feedback['back']['score'], feedback['back']['color'], feedback['back']['status'])

    # Feedback banner
    feedback_text = feedback.get('current_feedback_text', 'Analyzing...')
    text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    banner_y = h - 35
    cv2.rectangle(expanded_image, (0, banner_y), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(expanded_image[banner_y:h], 0.75, image[banner_y:h], 0.25, 0, expanded_image[banner_y:h])
    cv2.putText(expanded_image, feedback_text, (text_x, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback.get('current_feedback_color', (255, 255, 255)), 2)

    return expanded_image

class PostureCalibrationData:
    """Store calibration data for squat posture."""
    def __init__(self):
        self.calibrated = False
        self.reference_knee_angle = None
        self.reference_spine_dev = None
        self.calibration_factor = 1.0

    def calibrate(self, feedback):
        """Store reference values from a good squat position."""
        self.reference_knee_angle = feedback['detailed_metrics']['avg_knee_angle']
        self.reference_spine_dev = feedback['detailed_metrics']['avg_spine_dev']
        self.calibrated = True

    def get_calibration_data(self):
        """Return calibration data."""
        return {
            'calibrated': self.calibrated,
            'reference_knee_angle': self.reference_knee_angle,
            'reference_spine_dev': self.reference_spine_dev,
            'calibration_factor': self.calibration_factor
        }

class SquatCoach:
    """Main class for squat posture correction."""
    def __init__(self):
        self.calibration = PostureCalibrationData()
        self.current_state = 'standing'
        self.in_squat = False
        self.rep_count = 0
        self.feedback_history = deque(maxlen=30)
        self.last_feedback_time = 0
        self.feedback_interval = 3
        self.current_feedback = {'text': "Analyzing...", 'color': (255, 255, 255)}

    def update_feedback(self, feedback, state):
        """Update feedback based on squat analysis."""
        scores = {
            'knee': feedback['knee']['score'],
            'hip': feedback['hip']['score'],
            'back': feedback['back']['score'],
            'knee_pos': feedback['knee_pos']['score']
        }
        worst_aspect = min(scores, key=scores.get)

        if state == 'squatting' and scores[worst_aspect] < 70:
            self.feedback_history.append(worst_aspect)
            current_time = time.time()
            if current_time - self.last_feedback_time > self.feedback_interval:
                counts = {key: sum(1 for x in self.feedback_history if x == key) for key in scores}
                for aspect in ['hip', 'knee', 'back', 'knee_pos']:
                    if counts.get(aspect, 0) >= 5:
                        self.current_feedback = {
                            'text': feedback[aspect]['feedback'],
                            'color': feedback[aspect]['color']
                        }
                        self.last_feedback_time = current_time
                        break
        elif state == 'standing':
            self.current_feedback = {'text': "Stand tall, prepare for squat", 'color': (255, 255, 255)}
        elif scores[worst_aspect] >= 85:
            self.current_feedback = {'text': "Great squat form!", 'color': (0, 255, 0)}

    def run(self, camera_index=0, width=640, height=480):
        """Main execution loop."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1) as pose:
            is_calibration_mode = False
            calibration_countdown = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture video")
                    break
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    calibration_data = self.calibration.get_calibration_data() if self.calibration.calibrated else None
                    feedback = get_posture_feedback(landmarks, image.shape, calibration_data)

                    if is_calibration_mode:
                        if calibration_countdown > 0:
                            cv2.putText(image, f"Calibrating in {calibration_countdown}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            calibration_countdown -= 1
                        else:
                            self.calibration.calibrate(feedback)
                            is_calibration_mode = False
                            print("Calibration complete!")

                    if 'visibility_issue' in feedback:
                        cv2.putText(image, f"Missing: {feedback['missing_points']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # Update state
                        avg_knee_angle = feedback['detailed_metrics']['avg_knee_angle']
                        if avg_knee_angle > 160:
                            new_state = 'standing'
                        elif avg_knee_angle < 100:
                            new_state = 'squatting'
                        else:
                            new_state = 'transitioning'

                        # Rep counting
                        if self.current_state == 'standing' and new_state == 'squatting':
                            self.in_squat = True
                        elif self.current_state == 'squatting' and new_state == 'standing' and self.in_squat:
                            if feedback['hip']['score'] >= 60:  # Ensure sufficient depth
                                self.rep_count += 1
                            self.in_squat = False
                        self.current_state = new_state

                        # Update feedback
                        self.update_feedback(feedback, new_state)
                        feedback['current_feedback_text'] = self.current_feedback['text']
                        feedback['current_feedback_color'] = self.current_feedback['color']

                        # Visualize
                        image = draw_posture_visualization(image, feedback)
                        image = create_feedback_panel(image, feedback, new_state, self.rep_count)
                else:
                    cv2.putText(image, "No pose detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(image, "Press 'c' to calibrate, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow('SquatCoach', image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    is_calibration_mode = True
                    calibration_countdown = 3
                    print("Calibration mode activated. Perform a perfect squat.")

        cap.release()
        cv2.destroyAllWindows()

def main():
    """Run the SquatCoach application."""
    print("Starting SquatCoach...")
    print("Controls: Press 'c' to calibrate, 'q' to quit.")
    coach = SquatCoach()
    coach.run()
    print("SquatCoach closed.")

if __name__ == "__main__":
    main()
