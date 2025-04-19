import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize MediaPipe components once
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Vectorized angle calculation functions
def calculate_angle(a, b, c):
    """Vectorized angle calculation between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    
    # Normalize vectors for numerical stability
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    
    # Handle zero-length vectors
    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return 0.0
    
    # Vectorized dot product and angle calculation
    cosine = np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    
    return angle

def calculate_angle_with_vertical(point1, point2):
    """Optimized angle calculation between a line and the vertical axis."""
    vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    norm = np.linalg.norm(vector)
    
    # Early return for zero-length vectors
    if norm < 1e-6:
        return 0.0
    
    # Pre-compute the normalized vector and vertical vector
    vector = vector / norm
    vertical = np.array([0, -1])
    
    # Optimized angle calculation using dot product
    dot_product = np.dot(vector, vertical)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)

def get_computer_posture_feedback(landmarks, image_shape, calibration_data=None):
    """Optimized posture analysis function with pre-calculated values and reduced redundancy."""
    h, w = image_shape[0], image_shape[1]
    
    # Fixed visibility threshold
    visibility_threshold = 0.7
    
    # Pre-compute scaling multiplier for coordinate conversion
    scale = np.array([w, h])
    
    # Define key points dictionary - moved out of function scope
    key_points = {
        'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        'left_hip': mp_pose.PoseLandmark.LEFT_HIP.value,
        'right_hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
        'nose': mp_pose.PoseLandmark.NOSE.value,
        'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST.value,
        'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value,
        'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
        'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        'left_eye': mp_pose.PoseLandmark.LEFT_EYE.value,
        'right_eye': mp_pose.PoseLandmark.RIGHT_EYE.value
    }
    
    # Efficiently extract points
    points = {}
    for key, idx in key_points.items():
        landmark = landmarks[idx]
        if landmark.visibility < visibility_threshold:
            return {'visibility_issue': True, 'missing_points': key}
        points[key] = np.array([landmark.x, landmark.y]) * scale
    
    # Calculate midpoints efficiently
    points['shoulder_mid'] = (points['left_shoulder'] + points['right_shoulder']) * 0.5
    points['hip_mid'] = (points['left_hip'] + points['right_hip']) * 0.5
    points['eye_mid'] = (points['left_eye'] + points['right_eye']) * 0.5
    
    # Get calibration factor - default to 1.0 if not calibrated
    calibration_factor = calibration_data.get('calibration_factor', 1.0) if calibration_data else 1.0
    
    # Pre-compute thresholds with calibration applied
    # Use lookup tables for common thresholds
    thresholds = {
        'excellent': 5 * calibration_factor,
        'good': 10 * calibration_factor,
        'fair': 15 * calibration_factor
    }
    
    # 1. HEAD ANALYSIS - Screen height assessment
    head_tilt_angle = calculate_angle_with_vertical(points['eye_mid'], points['nose'])
    
    # Optimize with lookup dictionary for score assignment
    if head_tilt_angle <= thresholds['excellent']:
        head_results = {"status": "Excellent", "feedback": "Head is perfectly neutral", 
                       "color": (0, 255, 0), "score": 100}
    elif head_tilt_angle <= thresholds['good']:
        head_results = {"status": "Good", "feedback": "Head is slightly tilted", 
                       "color": (0, 255, 0), "score": 80}
    elif head_tilt_angle <= thresholds['fair']:
        head_results = {"status": "Fair", "feedback": "Reduce head tilt for better posture", 
                       "color": (0, 165, 255), "score": 60}
    else:
        head_results = {"status": "Poor", "feedback": "Head is significantly tilted; adjust screen height", 
                       "color": (0, 0, 255), "score": 30}
    
    head_results['angle'] = head_tilt_angle
    
    # 2. WRIST POSITION ANALYSIS
    # Pre-define reference points for angle calculation
    left_wrist_ref = np.array([points['left_wrist'][0] + 100, points['left_wrist'][1]])
    right_wrist_ref = np.array([points['right_wrist'][0] + 100, points['right_wrist'][1]])
    
    # Batch calculation for wrist angles
    left_forearm_angle = calculate_angle(points['left_elbow'], points['left_wrist'], left_wrist_ref)
    right_forearm_angle = calculate_angle(points['right_elbow'], points['right_wrist'], right_wrist_ref)
    
    # Optimize score calculation using numpy functions
    left_forearm_score = np.piecewise(
        np.abs(left_forearm_angle),
        [
            np.abs(left_forearm_angle) <= thresholds['excellent'],
            (np.abs(left_forearm_angle) > thresholds['excellent']) & (np.abs(left_forearm_angle) <= thresholds['good']),
            (np.abs(left_forearm_angle) > thresholds['good']) & (np.abs(left_forearm_angle) <= thresholds['fair'])
        ],
        [100, 80, 60, 30]
    )
    
    right_forearm_score = np.piecewise(
        np.abs(right_forearm_angle),
        [
            np.abs(right_forearm_angle) <= thresholds['excellent'],
            (np.abs(right_forearm_angle) > thresholds['excellent']) & (np.abs(right_forearm_angle) <= thresholds['good']),
            (np.abs(right_forearm_angle) > thresholds['good']) & (np.abs(right_forearm_angle) <= thresholds['fair'])
        ],
        [100, 80, 60, 30]
    )
    
    wrist_score = (left_forearm_score + right_forearm_score) / 2
    
    # Use score lookup for wrist status
    if wrist_score >= 90:
        wrist_results = {"status": "Excellent", "feedback": "Wrists are perfectly aligned", 
                        "color": (0, 255, 0), "score": wrist_score}
    elif wrist_score >= 70:
        wrist_results = {"status": "Good", "feedback": "Wrists are well positioned", 
                        "color": (0, 255, 0), "score": wrist_score}
    elif wrist_score >= 50:
        wrist_results = {"status": "Fair", "feedback": "Adjust wrist position for better alignment", 
                        "color": (0, 165, 255), "score": wrist_score}
    else:
        wrist_results = {"status": "Poor", "feedback": "Wrists are misaligned; ensure forearms are parallel to the desk", 
                        "color": (0, 0, 255), "score": wrist_score}
    
    # 3. SPINE ANALYSIS
    spine_angle = calculate_angle_with_vertical(points['hip_mid'], points['shoulder_mid'])
    
    # Simplified spine curvature calculation
    spine_curvature = 0  # Default value if no additional spine points
    
    # Lookup for spine status
    if spine_angle <= thresholds['excellent'] and spine_curvature < 10:
        spine_results = {"status": "Excellent", "feedback": "Spine is perfectly aligned", 
                        "color": (0, 255, 0), "score": 100}
    elif spine_angle <= thresholds['good'] and spine_curvature < 20:
        spine_results = {"status": "Good", "feedback": "Spine alignment is good", 
                        "color": (0, 255, 0), "score": 80}
    elif spine_angle <= thresholds['fair'] and spine_curvature < 35:
        spine_results = {"status": "Fair", "feedback": "Straighten your back slightly", 
                        "color": (0, 165, 255), "score": 60}
    else:
        spine_results = {"status": "Poor", "feedback": "Avoid slouching; maintain a natural S-curve", 
                        "color": (0, 0, 255), "score": 30}
    
    spine_results['angle'] = spine_angle
    
    # Calculate overall score with vectorized operations
    overall_score = int(0.4 * head_results['score'] + 0.3 * wrist_results['score'] + 0.3 * spine_results['score'])
    
    # Optimized overall status lookup
    if overall_score >= 90:
        overall_results = {"status": "Excellent", "score": overall_score, "color": (0, 255, 0)}
    elif overall_score >= 70:
        overall_results = {"status": "Good", "score": overall_score, "color": (0, 255, 0)}
    elif overall_score >= 50:
        overall_results = {"status": "Fair", "score": overall_score, "color": (0, 165, 255)}
    else:
        overall_results = {"status": "Poor", "score": overall_score, "color": (0, 0, 255)}
    
    # Return optimized result structure
    return {
        'head': head_results,
        'wrist': wrist_results,
        'spine': spine_results,
        'overall': overall_results,
        'points': points
    }

def draw_posture_visualization(image, feedback):
    """Optimized visualization function with fewer conversions."""
    points = feedback['points']
    
    # Pre-compute common points as integers to avoid repeated conversions
    hip_mid_int = (int(points['hip_mid'][0]), int(points['hip_mid'][1]))
    shoulder_mid_int = (int(points['shoulder_mid'][0]), int(points['shoulder_mid'][1]))
    eye_mid_int = (int(points['eye_mid'][0]), int(points['eye_mid'][1]))
    nose_int = (int(points['nose'][0]), int(points['nose'][1]))
    
    # Draw reference and posture lines
    cv2.line(image, hip_mid_int, (hip_mid_int[0], 0), (200, 200, 200), 1)
    cv2.line(image, shoulder_mid_int, hip_mid_int, feedback['spine']['color'], 3)
    cv2.line(image, eye_mid_int, nose_int, feedback['head']['color'], 3)
    
    # Draw wrist alignment using precomputed integers
    for side in ['left', 'right']:
        elbow_int = (int(points[f'{side}_elbow'][0]), int(points[f'{side}_elbow'][1]))
        wrist_int = (int(points[f'{side}_wrist'][0]), int(points[f'{side}_wrist'][1]))
        cv2.line(image, elbow_int, wrist_int, feedback['wrist']['color'], 2)
    
    # Add angle annotations
    nose_text_pos = (nose_int[0] + 10, nose_int[1])
    shoulder_text_pos = (shoulder_mid_int[0] + 10, shoulder_mid_int[1])
    
    # Format strings once
    head_angle_text = f"{feedback['head']['angle']:.1f}°"
    spine_angle_text = f"{feedback['spine']['angle']:.1f}°"
    
    cv2.putText(image, head_angle_text, nose_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, feedback['head']['color'], 1)
    cv2.putText(image, spine_angle_text, shoulder_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, feedback['spine']['color'], 1)
    
    return image

def create_feedback_panel(image, feedback):
    """Optimized feedback panel creation with pre-allocated memory."""
    h, w, c = image.shape
    panel_height = 100
    
    # Pre-allocate expanded image buffer with final dimensions
    expanded_image = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    
    # Copy original image and panel background in one operation each
    expanded_image[0:h, 0:w] = image
    expanded_image[h:h+panel_height, 0:w] = (18, 18, 24)
    
    # Pre-format text for feedback
    head_text = f"Head: {feedback['head']['status']} ({feedback['head']['angle']:.1f}°)"
    wrist_text = f"Wrists: {feedback['wrist']['status']}"
    spine_text = f"Spine: {feedback['spine']['status']} ({feedback['spine']['angle']:.1f}°)"
    overall_text = f"Overall: {feedback['overall']['status']} ({feedback['overall']['score']})"
    
    # Draw text with pre-computed positions
    cv2.putText(expanded_image, head_text, (20, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, feedback['head']['color'], 2)
    cv2.putText(expanded_image, wrist_text, (20, h + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, feedback['wrist']['color'], 2)
    cv2.putText(expanded_image, spine_text, (20, h + 90), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, feedback['spine']['color'], 2)
    cv2.putText(expanded_image, overall_text, (w - 200, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, feedback['overall']['color'], 2)
    
    # Feedback banner with optimized transparency overlay
    banner_height = 35
    banner_y = h - banner_height
    
    # Create banner in one operation
    overlay = np.zeros((banner_height, w, 3), dtype=np.uint8)
    overlay[:] = (15, 15, 25)
    
    # Use direct array assignment with alpha blending
    alpha = 0.75
    expanded_image[banner_y:h, 0:w] = cv2.addWeighted(
        overlay, alpha, 
        image[banner_y:h, 0:w], 1 - alpha, 
        0
    )
    
    # Add feedback text
    text = feedback.get('current_feedback_text', 'Analyzing...')
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(expanded_image, text, (text_x, h - banner_height + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback.get('current_feedback_color', (255, 255, 255)), 2)
    
    return expanded_image

class PostureCalibrationData:
    """Optimized calibration data storage."""
    __slots__ = ('calibrated', 'reference_head_angle', 'reference_spine_angle', 'calibration_factor')
    
    def __init__(self):
        self.calibrated = False
        self.reference_head_angle = None
        self.reference_spine_angle = None
        self.calibration_factor = 1.0
    
    def calibrate(self, feedback):
        """Store reference values from optimal posture."""
        self.reference_head_angle = feedback['head']['angle']
        self.reference_spine_angle = feedback['spine']['angle']
        self.calibrated = True
    
    def get_calibration_data(self):
        """Return calibration data as a dictionary."""
        # Return direct references instead of creating new dict
        if not self.calibrated:
            return None
        
        return {
            'calibrated': self.calibrated,
            'reference_head_angle': self.reference_head_angle,
            'reference_spine_angle': self.reference_spine_angle,
            'calibration_factor': self.calibration_factor
        }

class ComputerPostureCoach:
    """Optimized main class for posture detection."""
    def __init__(self):
        self.calibration = PostureCalibrationData()
        self.feedback_history = deque(maxlen=30)
        self.current_feedback = {'text': "Analyzing your posture...", 'color': (255, 255, 255)}
        self.last_feedback_time = 0
        self.feedback_interval = 5
        self.feedback_threshold = 5
        
        # Pre-compile common text and positions
        self.calibration_text = "Press 'c' to calibrate, 'q' to quit"
        self.no_pose_text = "No pose detected"
        
        # Pre-define common font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def run(self, camera_index=0, width=640, height=480):
        """Optimized main execution loop."""
        # Initialize camera once
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Configure MediaPipe pose detection with optimal settings
        pose_config = {
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.7,
            'model_complexity': 1,
            'static_image_mode': False,  # Dynamic tracking for video
        }
        
        with mp_pose.Pose(**pose_config) as pose:
            is_calibration_mode = False
            calibration_countdown = 0
            
            # Pre-allocate buffers for image processing
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Main loop
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture video")
                    break
                
                # Flip the frame horizontally for a mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert color space efficiently
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb_image)
                
                # Process image with MediaPipe
                rgb_image.flags.writeable = False
                results = pose.process(rgb_image)
                rgb_image.flags.writeable = True
                
                # Convert back to BGR for display
                image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Only get calibration data if calibrated
                    calibration_data = self.calibration.get_calibration_data() if self.calibration.calibrated else None
                    
                    # Handle calibration mode
                    if is_calibration_mode:
                        if calibration_countdown > 0:
                            cv2.putText(image, f"Calibrating in {calibration_countdown}...", 
                                       (50, 50), self.font, 1, (0, 0, 255), 2)
                            calibration_countdown -= 1
                        else:
                            # Get feedback for calibration
                            feedback = get_computer_posture_feedback(landmarks, image.shape)
                            self.calibration.calibrate(feedback)
                            is_calibration_mode = False
                            print("Calibration complete!")
                    else:
                        # Get feedback with calibration data if available
                        feedback = get_computer_posture_feedback(landmarks, image.shape, calibration_data)
                        
                        if 'visibility_issue' in feedback:
                            visibility_msg = f"Cannot see full body: {feedback['missing_points']} not visible"
                            cv2.putText(image, visibility_msg, (20, 60), self.font, 0.7, (0, 0, 255), 2)
                        else:
                            # Draw visualization and create feedback panel
                            image = draw_posture_visualization(image, feedback)
                            self.update_feedback(feedback)
                            feedback['current_feedback_text'] = self.current_feedback['text']
                            feedback['current_feedback_color'] = self.current_feedback['color']
                            image = create_feedback_panel(image, feedback)
                else:
                    # Show message when no pose is detected
                    cv2.putText(image, self.no_pose_text, (20, 60), self.font, 0.7, (0, 0, 255), 2)
                
                # Add instructions text
                cv2.putText(image, self.calibration_text, (10, 30), self.font, 0.6, (255, 255, 255), 1)
                
                # Show the image
                cv2.imshow('ComputerPostureCoach', image)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    is_calibration_mode = True
                    calibration_countdown = 3
                    print("Calibration mode activated. Assume optimal posture.")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def update_feedback(self, feedback):
        """Optimized feedback update algorithm."""
        # Get all scores
        scores = {
            'head': feedback['head']['score'], 
            'wrist': feedback['wrist']['score'], 
            'spine': feedback['spine']['score']
        }
        
        # Find the worst aspect efficiently
        worst_aspect = min(scores, key=scores.get)
        
        # Only update feedback when necessary
        if scores[worst_aspect] < 70:
            self.feedback_history.append(worst_aspect)
            
            # Use collections.Counter for efficient counting
            from collections import Counter
            counts = Counter(self.feedback_history)
            
            # Check if it's time to update feedback
            current_time = time.time()
            if current_time - self.last_feedback_time > self.feedback_interval:
                # Find most common issue above threshold
                aspect_to_update = None
                for aspect, count in counts.items():
                    if count >= self.feedback_threshold:
                        aspect_to_update = aspect
                        break
                
                if aspect_to_update:
                    self.current_feedback = {
                        'text': feedback[aspect_to_update]['feedback'], 
                        'color': feedback[aspect_to_update]['color']
                    }
                    self.last_feedback_time = current_time
        
        elif feedback['overall']['score'] >= 85:
            # Positive feedback for good posture
            self.current_feedback = {
                'text': "Great computer posture! Keep it up!", 
                'color': (0, 255, 0)
            }

def main():
    """Optimized main function."""
    print("Starting ComputerPostureCoach...")
    coach = ComputerPostureCoach()
    
    try:
        coach.run()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("ComputerPostureCoach closed.")

if __name__ == "__main__":
    main()
