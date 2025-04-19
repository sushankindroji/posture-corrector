import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def calculate_angle_with_vertical(point1, point2):
    """Calculate angle between a line and the vertical axis."""
    vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    if np.linalg.norm(vector) < 1e-6:
        return 0.0
    vector = vector / np.linalg.norm(vector)
    vertical = np.array([0, -1])
    dot_product = np.dot(vector, vertical)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)

def get_deadlift_feedback(landmarks, image_shape, previous_hip=None, previous_shoulder=None):
    """Analyze deadlift posture and provide feedback."""
    h, w = image_shape[0], image_shape[1]
    feedback = {}

    # Extract key points (left side for side view, assuming facing right)
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h])
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h])
    left_ear = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h])
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h])
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h])

    # Calculate angles
    spine_angle = calculate_angle(left_hip, left_shoulder, left_ear)
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    torso_angle = calculate_angle_with_vertical(left_hip, left_shoulder)

    # Compute movement deltas
    delta_y_hip = delta_y_shoulder = None
    if previous_hip is not None and previous_shoulder is not None:
        delta_y_hip = left_hip[1] - previous_hip[1]
        delta_y_shoulder = left_shoulder[1] - previous_shoulder[1]

    # Spine neutrality
    if 170 <= spine_angle <= 190:
        feedback['spine'] = {'status': 'Good', 'feedback': 'Back is neutral', 'color': (0, 255, 0), 'score': 100}
    elif 160 <= spine_angle < 170:
        feedback['spine'] = {'status': 'Fair', 'feedback': 'Slightly rounded back', 'color': (0, 165, 255), 'score': 60}
    elif spine_angle < 160:
        feedback['spine'] = {'status': 'Poor', 'feedback': 'Avoid rounding your back', 'color': (0, 0, 255), 'score': 30}
    elif 190 < spine_angle <= 200:
        feedback['spine'] = {'status': 'Fair', 'feedback': 'Slightly arched back', 'color': (0, 165, 255), 'score': 60}
    else:
        feedback['spine'] = {'status': 'Poor', 'feedback': 'Avoid arching your back', 'color': (0, 0, 255), 'score': 30}

    # Knee bend and shoulder position at bottom
    if torso_angle > 50:
        # Knee bend
        if 120 <= knee_angle <= 150:
            feedback['knee'] = {'status': 'Good', 'feedback': 'Good knee bend', 'color': (0, 255, 0), 'score': 100}
        elif knee_angle < 120:
            feedback['knee'] = {'status': 'Poor', 'feedback': 'Too much knee bend', 'color': (0, 0, 255), 'score': 30}
        else:
            feedback['knee'] = {'status': 'Poor', 'feedback': 'Too little knee bend', 'color': (0, 0, 255), 'score': 30}

        # Shoulder position
        if left_shoulder[0] < left_hip[0]:
            feedback['shoulder'] = {'status': 'Good', 'feedback': 'Shoulders in good position', 'color': (0, 255, 0), 'score': 100}
        else:
            feedback['shoulder'] = {'status': 'Poor', 'feedback': 'Move shoulders forward', 'color': (0, 0, 255), 'score': 30}

    # Hip and shoulder movement during lift
    if delta_y_hip is not None and delta_y_hip < 0:
        if delta_y_hip >= delta_y_shoulder * 1.2:
            feedback['movement'] = {'status': 'Good', 'feedback': 'Good lift technique', 'color': (0, 255, 0), 'score': 100}
        else:
            feedback['movement'] = {'status': 'Poor', 'feedback': 'Hips rising too fast', 'color': (0, 0, 255), 'score': 30}

    # Overall score
    scores = [f['score'] for f in feedback.values()]
    overall_score = sum(scores) / len(scores) if scores else 0
    feedback['overall'] = {
        'score': int(overall_score),
        'status': 'Good' if overall_score > 80 else 'Fair' if overall_score > 50 else 'Poor',
        'color': (0, 255, 0) if overall_score > 80 else (0, 165, 255) if overall_score > 50 else (0, 0, 255)
    }

    feedback['points'] = {
        'left_hip': left_hip, 'left_shoulder': left_shoulder, 'left_ear': left_ear,
        'left_knee': left_knee, 'left_ankle': left_ankle
    }
    feedback['metrics'] = {'spine_angle': spine_angle, 'knee_angle': knee_angle, 'hip_angle': hip_angle, 'torso_angle': torso_angle}
    return feedback

def draw_deadlift_visualization(image, feedback):
    """Draw visual cues on the image."""
    points = feedback['points']
    metrics = feedback['metrics']

    # Draw spine line
    cv2.line(image, tuple(points['left_hip'].astype(int)), tuple(points['left_shoulder'].astype(int)),
             feedback['spine']['color'], 3, cv2.LINE_AA)
    cv2.line(image, tuple(points['left_shoulder'].astype(int)), tuple(points['left_ear'].astype(int)),
             feedback['spine']['color'], 3, cv2.LINE_AA)

    # Draw leg line if knee feedback present
    if 'knee' in feedback:
        cv2.line(image, tuple(points['left_hip'].astype(int)), tuple(points['left_knee'].astype(int)),
                 feedback['knee']['color'], 3, cv2.LINE_AA)
        cv2.line(image, tuple(points['left_knee'].astype(int)), tuple(points['left_ankle'].astype(int)),
                 feedback['knee']['color'], 3, cv2.LINE_AA)

    # Draw key points
    for point in points.values():
        cv2.circle(image, tuple(point.astype(int)), 4, (255, 255, 0), -1, cv2.LINE_AA)

    # Annotate angles
    cv2.putText(image, f"Spine: {metrics['spine_angle']:.1f}°",
                (int(points['left_shoulder'][0]) + 10, int(points['left_shoulder'][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['spine']['color'], 1, cv2.LINE_AA)
    if 'knee' in feedback:
        cv2.putText(image, f"Knee: {metrics['knee_angle']:.1f}°",
                    (int(points['left_knee'][0]) + 10, int(points['left_knee'][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['knee']['color'], 1, cv2.LINE_AA)

    return image

def create_feedback_panel(image, feedback):
    """Create a feedback panel below the image."""
    h, w, c = image.shape
    panel_height = 130
    expanded_image = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    expanded_image[0:h, 0:w] = image
    expanded_image[h:h+panel_height, 0:w] = (18, 18, 24)

    # Score text
    score_text = f"Form Score: {feedback['overall']['score']}"
    cv2.rectangle(expanded_image, (15, h + 15), (235, h + 45), (30, 30, 40), -1, cv2.LINE_AA)
    cv2.rectangle(expanded_image, (15, h + 15), (235, h + 45), feedback['overall']['color'], 1, cv2.LINE_AA)
    cv2.putText(expanded_image, score_text, (25, h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                feedback['overall']['color'], 2, cv2.LINE_AA)

    # Feedback bars
    bar_y = h + 55
    bar_spacing = 25
    bar_max_width = w - 240

    def draw_bar(y_pos, label, score, color, status):
        cv2.putText(expanded_image, label, (20, y_pos + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(expanded_image, (90, y_pos), (90 + bar_max_width, y_pos + 18), (35, 35, 45), -1, cv2.LINE_AA)
        bar_width = int((score / 100) * bar_max_width)
        cv2.rectangle(expanded_image, (90, y_pos), (90 + bar_width, y_pos + 18), color, -1, cv2.LINE_AA)
        cv2.putText(expanded_image, status, (100 + bar_max_width, y_pos + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)

    aspects = ['spine', 'knee', 'shoulder', 'movement']
    for i, aspect in enumerate(aspects):
        if aspect in feedback:
            draw_bar(bar_y + i * bar_spacing, f"{aspect.capitalize()}:", feedback[aspect]['score'],
                     feedback[aspect]['color'], feedback[aspect]['status'])

    # Footer
    cv2.putText(expanded_image, "DeadliftCoach | Press 'q' to exit", (w - 230, h + panel_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 150), 1, cv2.LINE_AA)

    return expanded_image

class DeadliftCoach:
    """Main class for deadlift posture correction."""
    def __init__(self):
        self.previous_left_hip = None
        self.previous_left_shoulder = None

    def run(self, camera_index=0, width=640, height=480):
        """Run the deadlift posture correction loop."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1) as pose:
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
                    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height])
                    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height])

                    feedback = get_deadlift_feedback(landmarks, image.shape, self.previous_left_hip, self.previous_left_shoulder)
                    self.previous_left_hip = left_hip
                    self.previous_left_shoulder = left_shoulder

                    image = draw_deadlift_visualization(image, feedback)
                    image = create_feedback_panel(image, feedback)
                else:
                    cv2.putText(image, "No pose detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(image, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('DeadliftCoach', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the DeadliftCoach."""
    print("Starting DeadliftCoach...")
    print("Initializing camera and models...")
    coach = DeadliftCoach()
    print("Ready! Starting deadlift posture detection.")
    coach.run()
    print("DeadliftCoach closed.")

if __name__ == "__main__":
    main()
