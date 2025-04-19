import cv2
import mediapipe as mp
import numpy as np
from time import time
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def get_pushup_feedback(landmarks, image_shape):
    h, w = image_shape[:2]
    visibility_threshold = 0.7
    
    key_landmarks = {
        'left': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, 
                 mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, 
                 mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE],
        'right': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, 
                  mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_HIP, 
                  mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    }

    points = {'left': {}, 'right': {}}
    visible_sides = []
    
    for side, landmarks_list in key_landmarks.items():
        all_visible = True
        for landmark in landmarks_list:
            lm = landmarks[landmark.value]
            if lm.visibility < visibility_threshold:
                all_visible = False
                break
            points[side][landmark.name] = np.array([lm.x * w, lm.y * h])
        if all_visible:
            visible_sides.append(side)

    if not visible_sides:
        return {'visibility_issue': True, 'message': 'Adjust position - key points not visible'}

    if len(visible_sides) == 1:
        side = visible_sides[0]
    else:
        left_vis = sum(landmarks[lm.value].visibility for lm in key_landmarks['left']) / len(key_landmarks['left'])
        right_vis = sum(landmarks[lm.value].visibility for lm in key_landmarks['right']) / len(key_landmarks['right'])
        side = 'left' if left_vis >= right_vis else 'right'
    
    avg_visibility = sum(landmarks[lm.value].visibility for lm in key_landmarks[side]) / len(key_landmarks[side])
    
    p = points[side]
    prefix = f"{side.upper()}_"

    elbow_angle = calculate_angle(p[f'{prefix}SHOULDER'], p[f'{prefix}ELBOW'], p[f'{prefix}WRIST'])
    back_angle = calculate_angle(p[f'{prefix}SHOULDER'], p[f'{prefix}HIP'], p[f'{prefix}KNEE'])
    hip_angle = calculate_angle(p[f'{prefix}SHOULDER'], p[f'{prefix}HIP'], p[f'{prefix}ANKLE'])
    knee_angle = calculate_angle(p[f'{prefix}HIP'], p[f'{prefix}KNEE'], p[f'{prefix}ANKLE'])

    position = 'transition'
    if elbow_angle < 85:
        position = 'bottom'
    elif elbow_angle > 165:
        position = 'top'

    feedback = {'corrections': [], 'score': 100, 'strengths': []}

    if position == 'bottom':
        if elbow_angle > 100:
            feedback['corrections'].append('Lower body more (elbows below 90 degrees)')
            feedback['score'] -= 30
        elif elbow_angle > 90:
            feedback['corrections'].append('Lower body slightly further')
            feedback['score'] -= 15
        else:
            feedback['strengths'].append('Excellent depth achieved')
    elif position == 'top':
        if elbow_angle < 150:
            feedback['corrections'].append('Fully extend arms')
            feedback['score'] -= 25
        elif elbow_angle < 165:
            feedback['corrections'].append('Complete arm extension')
            feedback['score'] -= 10
        else:
            feedback['strengths'].append('Perfect arm extension')

    if 178 <= back_angle <= 182:
        feedback['strengths'].append('Perfect back alignment')
    elif 175 <= back_angle <= 185:
        feedback['strengths'].append('Good back alignment')
    else:
        if back_angle < 175:
            feedback['corrections'].append('Back sagging - engage core')
            feedback['score'] -= 30
        else:
            feedback['corrections'].append('Back arching - lower hips')
            feedback['score'] -= 30

    if 175 <= hip_angle <= 185:
        feedback['strengths'].append('Ideal hip position')
    elif 170 <= hip_angle <= 190:
        pass
    else:
        feedback['corrections'].append('Adjust hip position - maintain straight body')
        feedback['score'] -= 20

    if 165 <= knee_angle <= 185:
        feedback['strengths'].append('Good leg position')
    elif 160 <= knee_angle <= 190:
        pass
    else:
        feedback['corrections'].append('Keep legs straighter')
        feedback['score'] -= 15

    if feedback['score'] >= 95:
        feedback['grade'] = 'Perfect'
        feedback['grade_color'] = (0, 255, 0)
    elif feedback['score'] >= 85:
        feedback['grade'] = 'Excellent'
        feedback['grade_color'] = (0, 255, 128)
    elif feedback['score'] >= 70:
        feedback['grade'] = 'Good'
        feedback['grade_color'] = (0, 255, 255)
    elif feedback['score'] >= 50:
        feedback['grade'] = 'Fair'
        feedback['grade_color'] = (0, 165, 255)
    else:
        feedback['grade'] = 'Needs Improvement'
        feedback['grade_color'] = (0, 0, 255)

    feedback['score'] = max(0, feedback['score'])
    back_color = (0, 255, 0) if 175 <= back_angle <= 185 else (0, 0, 255)
    
    feedback['confidence'] = min(100, int(avg_visibility * 100))

    return {
        'position': position,
        'feedback': feedback,
        'elbow_angle': elbow_angle,
        'back_angle': back_angle,
        'hip_angle': hip_angle,
        'knee_angle': knee_angle,
        'points': p,
        'back_color': back_color,
        'side': side,
        'prefix': prefix
    }

def draw_visualization(image, feedback):
    if 'visibility_issue' in feedback:
        return image

    p = feedback['points']
    back_color = feedback['back_color']
    prefix = feedback['prefix']

    cv2.line(image, tuple(p[f'{prefix}SHOULDER'].astype(int)),
             tuple(p[f'{prefix}ELBOW'].astype(int)), (255, 0, 0), 3)
    cv2.line(image, tuple(p[f'{prefix}ELBOW'].astype(int)),
             tuple(p[f'{prefix}WRIST'].astype(int)), (255, 0, 0), 3)

    cv2.line(image, tuple(p[f'{prefix}SHOULDER'].astype(int)),
             tuple(p[f'{prefix}HIP'].astype(int)), back_color, 3)
    
    hip_knee_color = (0, 255, 0) if 170 <= feedback['hip_angle'] <= 190 else (0, 165, 255)
    knee_ankle_color = (0, 255, 0) if 165 <= feedback['knee_angle'] <= 185 else (0, 165, 255)
    
    cv2.line(image, tuple(p[f'{prefix}HIP'].astype(int)),
             tuple(p[f'{prefix}KNEE'].astype(int)), hip_knee_color, 3)
    cv2.line(image, tuple(p[f'{prefix}KNEE'].astype(int)),
             tuple(p[f'{prefix}ANKLE'].astype(int)), knee_ankle_color, 3)

    for point_name, point in p.items():
        if point_name in [f'{prefix}SHOULDER', f'{prefix}ELBOW', f'{prefix}WRIST', 
                          f'{prefix}HIP', f'{prefix}KNEE', f'{prefix}ANKLE']:
            cv2.circle(image, tuple(point.astype(int)), 8, (255, 255, 0), -1)

    cv2.putText(image, f"Elbow: {feedback['elbow_angle']:.1f}°",
                (int(p[f'{prefix}ELBOW'][0]) + 10, int(p[f'{prefix}ELBOW'][1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(image, f"Back: {feedback['back_angle']:.1f}°",
                (int(p[f'{prefix}HIP'][0]) + 10, int(p[f'{prefix}HIP'][1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def create_feedback_panel(image, feedback, pushup_count, streak_data=None):
    h, w = image.shape[:2]
    panel_height = 160
    expanded = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    expanded[:h, :] = image

    # Gradient background for panel
    for i in range(panel_height):
        alpha = i / panel_height
        color = (int(30 + alpha * 20), int(30 + alpha * 20), int(40 + alpha * 30))
        expanded[h + i, :] = color

    # Header bar
    cv2.rectangle(expanded, (0, h), (w, h + 40), (40, 40, 60), -1)
    cv2.line(expanded, (0, h + 40), (w, h + 40), (100, 100, 120), 1)

    # Push-up count
    cv2.putText(expanded, f"Push-ups: {pushup_count}", (20, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Streak data
    if streak_data:
        cv2.putText(expanded, f"Best Streak: {streak_data['best']}", (w - 220, h + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        if 'avg_score' in streak_data:
            avg_score_color = (0, 255, 0) if streak_data['avg_score'] >= 90 else \
                            (0, 255, 255) if streak_data['avg_score'] >= 70 else (0, 165, 255)
            cv2.putText(expanded, f"Average Score: {streak_data['avg_score']}%", (w - 400, h + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, avg_score_color, 2)

    # Feedback content
    if 'visibility_issue' in feedback:
        cv2.putText(expanded, feedback['message'], (w // 3, h + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    else:
        # Position indicator
        position_text = f"Position: {feedback['position'].upper()}"
        position_color = (0, 165, 255) if feedback['position'] == 'bottom' else \
                        (0, 255, 0) if feedback['position'] == 'top' else (0, 255, 255)
        cv2.putText(expanded, position_text, (20, h + 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, position_color, 2)

        # Scorecard
        score_box = np.zeros((70, 200, 3), dtype=np.uint8)
        for i in range(70):
            alpha = i / 70
            score_box[i, :] = (int(50 + alpha * 20), int(50 + alpha * 20), int(60 + alpha * 30))
        cv2.rectangle(score_box, (0, 0), (199, 69), (100, 100, 120), 1)
        expanded[h + 50:h + 120, w - 220:w - 20] = score_box

        cv2.putText(expanded, f"Score: {feedback['feedback']['score']}%", (w - 210, h + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(expanded, f"Grade: {feedback['feedback']['grade']}", (w - 210, h + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback['feedback']['grade_color'], 2)

        # Feedback columns
        cv2.putText(expanded, "Strengths", (20, h + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(expanded, "Corrections", (w // 2, h + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        for i, strength in enumerate(feedback['feedback']['strengths'][:2]):
            cv2.putText(expanded, strength, (20, h + 110 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        for i, correction in enumerate(feedback['feedback']['corrections'][:2]):
            cv2.putText(expanded, correction, (w // 2, h + 110 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

    return expanded

class PushupCoach:
    def __init__(self):
        self.pushup_count = 0
        self.previous_position = None
        self.position_history = []
        self.streak_data = {'current': 0, 'best': 0, 'scores': deque(maxlen=5)}
        self.rep_in_progress = False
        self.rep_timer = 0
        self.last_rep_time = 0
        self.form_history = deque(maxlen=3)

    def update_streak(self, score):
        self.streak_data['scores'].append(score)
        self.form_history.append(score)
        
        if score >= 70:
            self.streak_data['current'] += 1
            self.streak_data['best'] = max(self.streak_data['best'], self.streak_data['current'])
        else:
            self.streak_data['current'] = 0
        
        if self.streak_data['scores']:
            self.streak_data['avg_score'] = int(sum(self.streak_data['scores']) / len(self.streak_data['scores']))

    def run(self, camera_index=0, width=640, height=480):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=2) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Camera error")
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    feedback = get_pushup_feedback(results.pose_landmarks.landmark, image.shape)
                    
                    if not feedback.get('visibility_issue'):
                        current_position = feedback['position']
                        
                        if len(self.position_history) >= 5:
                            self.position_history.pop(0)
                        self.position_history.append(current_position)
                        
                        if not self.rep_in_progress and current_position == 'top':
                            self.rep_in_progress = True
                            self.rep_timer = time()
                        
                        if (self.rep_in_progress and 
                            current_position == 'top' and 
                            'bottom' in self.position_history[-5:] and
                            time() - self.rep_timer > 0.7):
                            
                            self.pushup_count += 1
                            self.update_streak(feedback['feedback']['score'])
                            self.rep_in_progress = False
                            self.last_rep_time = time()
                            
                        image = draw_visualization(image, feedback)
                        self.previous_position = current_position
                else:
                    feedback = {'visibility_issue': True, 'message': 'No pose detected'}

                if self.last_rep_time > 0 and time() - self.last_rep_time < 1.2:
                    counter_size = max(1.0, 1.5 - (time() - self.last_rep_time))
                    cv2.putText(image, f"+1", (width//2 - 20, height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, counter_size, (0, 255, 255), 2)

                image = create_feedback_panel(image, feedback, self.pushup_count, self.streak_data)
                
                cv2.putText(image, "Press 'q' to quit", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow('Pushup Coach', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Starting Pushup Coach...")
    print("Press 'q' to quit")
    coach = PushupCoach()
    coach.run()

if __name__ == "__main__":
    main()
