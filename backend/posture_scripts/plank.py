import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculate angle between three points in degrees (0-180°)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def calculate_score(value, optimal, max_deviation):
    """Convert a deviation into a score from 0 to 100."""
    deviation = abs(value - optimal)
    if deviation > max_deviation:
        return 0
    return int(100 * (1 - deviation / max_deviation))

def get_plank_feedback(landmarks, image_shape):
    """Analyze plank posture and return feedback with scores."""
    h, w = image_shape[0], image_shape[1]

    # Determine visible side
    left_vis = sum(landmarks[idx].visibility for idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                                         mp_pose.PoseLandmark.LEFT_HIP.value,
                                                         mp_pose.PoseLandmark.LEFT_ANKLE.value]) / 3
    right_vis = sum(landmarks[idx].visibility for idx in [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                                          mp_pose.PoseLandmark.RIGHT_HIP.value,
                                                          mp_pose.PoseLandmark.RIGHT_ANKLE.value]) / 3
    side = 'left' if left_vis > right_vis else 'right'
    indices = {
        'ankle': getattr(mp_pose.PoseLandmark, f'{side.upper()}_ANKLE').value,
        'knee': getattr(mp_pose.PoseLandmark, f'{side.upper()}_KNEE').value,
        'hip': getattr(mp_pose.PoseLandmark, f'{side.upper()}_HIP').value,
        'shoulder': getattr(mp_pose.PoseLandmark, f'{side.upper()}_SHOULDER').value,
        'elbow': getattr(mp_pose.PoseLandmark, f'{side.upper()}_ELBOW').value,
        'wrist': getattr(mp_pose.PoseLandmark, f'{side.upper()}_WRIST').value,
        'ear': getattr(mp_pose.PoseLandmark, f'{side.upper()}_EAR').value
    }

    # Check visibility
    if any(landmarks[idx].visibility < 0.5 for idx in indices.values()):
        return {'visibility_issue': True, 'feedback': 'Please ensure your full body is visible'}

    # Extract pixel coordinates
    points = {key: np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for key, idx in indices.items()}

    # Determine plank type
    elbow_angle = calculate_angle(points['shoulder'], points['elbow'], points['wrist'])
    if elbow_angle > 150:
        plank_type = 'straight_arm'
    elif elbow_angle < 120:
        plank_type = 'forearm'
    else:
        return {'plank_type': 'unclear', 'feedback': 'Adjust your arms for a clear plank'}

    # Calculate angles and positions
    leg_angle = calculate_angle(points['ankle'], points['knee'], points['hip'])
    torso_expected_y = points['ankle'][1] + (points['shoulder'][1] - points['ankle'][1]) * \
                      (points['hip'][0] - points['ankle'][0]) / (points['shoulder'][0] - points['ankle'][0])
    torso_deviation = points['hip'][1] - torso_expected_y  # Positive = sagging, negative = piking
    head_deviation = points['ear'][1] - points['shoulder'][1]  # Positive = dropped, negative = raised

    # Arm alignment
    arm_x_dev = (abs(points['elbow'][0] - points['shoulder'][0]) if plank_type == 'forearm' else
                 abs(points['wrist'][0] - points['shoulder'][0]))
    arm_y_dev = (abs(points['elbow'][1] - points['ankle'][1]) if plank_type == 'forearm' else
                 abs(points['wrist'][1] - points['ankle'][1]))

    # Scores
    scores = {
        'legs': calculate_score(leg_angle, 180, 30),
        'torso': calculate_score(torso_deviation, 0, 0.1 * h),
        'head': calculate_score(head_deviation, 0, 0.1 * h),
        'arms': calculate_score(arm_x_dev, 0, 0.1 * w) * 0.7 + calculate_score(arm_y_dev, 0, 0.15 * h) * 0.3
    }

    # Feedback
    feedback = []
    if scores['legs'] < 80:
        feedback.append("Straighten your legs")
    if scores['torso'] < 80:
        feedback.append("Lift your hips" if torso_deviation > 0 else "Lower your hips")
    if scores['head'] < 80:
        feedback.append("Lift your head" if head_deviation > 0 else "Lower your head")
    if scores['arms'] < 80:
        feedback.append(f"Move your {'elbows' if plank_type == 'forearm' else 'hands'} under your shoulders")
        if (arm_y_dev > 0.15 * h):
            feedback.append(f"Ensure your {'elbows' if plank_type == 'forearm' else 'hands'} are on the ground")

    if not feedback:
        feedback.append("Perfect plank, bro!")

    return {
        'plank_type': plank_type,
        'scores': scores,
        'feedback': feedback,
        'points': points
    }

def draw_plank_visualization(image, feedback):
    """Draw posture lines, points, scorecard, and feedback."""
    h, w, c = image.shape
    points = feedback['points']
    scores = feedback['scores']

    # Colors based on scores
    colors = {key: (0, 255, 0) if scores[key] >= 80 else (0, 0, 255) for key in scores}

    # Draw body lines
    cv2.line(image, tuple(points['ankle'].astype(int)), tuple(points['knee'].astype(int)), colors['legs'], 2)
    cv2.line(image, tuple(points['knee'].astype(int)), tuple(points['hip'].astype(int)), colors['legs'], 2)
    cv2.line(image, tuple(points['hip'].astype(int)), tuple(points['shoulder'].astype(int)), colors['torso'], 2)
    cv2.line(image, tuple(points['shoulder'].astype(int)), tuple(points['ear'].astype(int)), colors['head'], 2)

    # Draw arm lines
    if feedback['plank_type'] == 'forearm':
        cv2.line(image, tuple(points['shoulder'].astype(int)), tuple(points['elbow'].astype(int)), colors['arms'], 2)
        cv2.line(image, tuple(points['elbow'].astype(int)), tuple(points['wrist'].astype(int)), (200, 200, 200), 1)
    else:
        cv2.line(image, tuple(points['shoulder'].astype(int)), tuple(points['elbow'].astype(int)), colors['arms'], 2)
        cv2.line(image, tuple(points['elbow'].astype(int)), tuple(points['wrist'].astype(int)), colors['arms'], 2)

    # Vertical reference line
    shoulder_x = int(points['shoulder'][0])
    cv2.line(image, (shoulder_x, 0), (shoulder_x, h), (150, 150, 150), 1)

    # Key points
    for pt in points.values():
        cv2.circle(image, tuple(pt.astype(int)), 4, (255, 255, 0), -1)

    # Scorecard (like sitting.py)
    panel_x, panel_y = w - 250, 20
    cv2.rectangle(image, (panel_x - 10, panel_y - 10), (panel_x + 240, panel_y + 160), (50, 50, 50), -1)
    cv2.putText(image, "Plank Scorecard", (panel_x, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for i, (part, score) in enumerate(scores.items()):
        label = part.capitalize()
        bar_length = int(score * 2)
        color = (0, 255, 0) if score >= 80 else (0, 0, 255)
        cv2.putText(image, f"{label}: {score}", (panel_x, panel_y + 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(image, (panel_x + 60, panel_y + 40 + i * 30), (panel_x + 60 + bar_length, panel_y + 50 + i * 30), color, -1)

    return image

class PlankCoach:
    """Real-time plank posture coach with scorecard."""
    def __init__(self):
        self.feedback_history = deque(maxlen=30)
        self.last_feedback_time = 0
        self.feedback_interval = 2  # Update every 2 seconds

    def run(self, camera_index=0, width=1280, height=720):
        """Run the plank coach."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Camera error, bro!")
                    break

                image = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    feedback = get_plank_feedback(results.pose_landmarks.landmark, image.shape)
                    if 'visibility_issue' in feedback:
                        cv2.putText(image, feedback['feedback'], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif feedback['plank_type'] == 'unclear':
                        cv2.putText(image, feedback['feedback'], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        image = draw_plank_visualization(image, feedback)
                        # Display feedback
                        for i, fb in enumerate(feedback['feedback']):
                            cv2.putText(image, fb, (20, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "No pose detected, bro!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Instructions
                cv2.putText(image, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("PlankCoach", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

def main():
    """Kick off the PlankCoach."""
    print("Starting PlankCoach, bro!")
    coach = PlankCoach()
    print("Get your camera on your side and plank like a champ!")
    coach.run()
    print("Done, bro! Keep planking!")

if __name__ == "__main__":
    main()
