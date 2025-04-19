import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points with higher precision."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def get_vertical_vector():
    """Return a unit vector pointing upward."""
    return np.array([0, -1])

def calculate_angle_with_vertical(point1, point2):
    """Calculate angle between a line (defined by two points) and the vertical axis."""
    vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    
    # Prevent division by zero
    if np.linalg.norm(vector) < 1e-6:
        return 0.0
    
    # Normalize vectors
    vector = vector / np.linalg.norm(vector)
    vertical = get_vertical_vector()
    
    # Calculate the angle using the dot product
    dot_product = np.dot(vector, vertical)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_side_lean(left_shoulder, right_shoulder, left_hip, right_hip):
    """Calculate if the person is leaning to one side."""
    # Calculate the angle between the line connecting shoulders and the line connecting hips
    shoulders_vector = np.array([right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]])
    hips_vector = np.array([right_hip[0] - left_hip[0], right_hip[1] - left_hip[1]])
    
    # Normalize vectors
    if np.linalg.norm(shoulders_vector) < 1e-6 or np.linalg.norm(hips_vector) < 1e-6:
        return 0.0
    
    shoulders_vector = shoulders_vector / np.linalg.norm(shoulders_vector)
    hips_vector = hips_vector / np.linalg.norm(hips_vector)
    
    # Calculate the angle
    dot_product = np.dot(shoulders_vector, hips_vector)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def check_rotation(left_shoulder, right_shoulder, left_hip, right_hip, image_width):
    """Detect if the person is rotated in their chair (not facing the camera directly)."""
    # Calculate expected width ratio between shoulders and hips
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    
    if hip_width < 1e-6:
        return 0, 1.0  # Avoid division by zero
    
    width_ratio = shoulder_width / hip_width
    
    # Calculate depth difference between left and right shoulder
    left_shoulder_z = left_shoulder[0] / image_width  # Use x-coordinate as proxy for depth
    right_shoulder_z = right_shoulder[0] / image_width
    
    rotation_factor = abs(left_shoulder_z - right_shoulder_z)
    
    return rotation_factor, width_ratio

def get_posture_feedback(landmarks, image_shape, calibration_data=None):
    """
    Enhanced posture analysis with more comprehensive metrics.
    
    Parameters:
    landmarks - MediaPipe pose landmarks
    image_shape - Dimensions of the image (height, width)
    calibration_data - Optional reference data from calibration phase
    
    Returns:
    Dictionary with posture analysis and feedback
    """
    h, w = image_shape[0], image_shape[1]
    feedback = {}
    
    # Extract key points with visibility check
    visibility_threshold = 0.65
    key_points = {
        'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        'left_hip': mp_pose.PoseLandmark.LEFT_HIP.value,
        'right_hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
        'left_ear': mp_pose.PoseLandmark.LEFT_EAR.value,
        'right_ear': mp_pose.PoseLandmark.RIGHT_EAR.value,
        'left_eye': mp_pose.PoseLandmark.LEFT_EYE.value,
        'right_eye': mp_pose.PoseLandmark.RIGHT_EYE.value,
        'nose': mp_pose.PoseLandmark.NOSE.value
    }
    
    points = {}
    
    # Check if critical landmarks are visible
    for key, landmark_idx in key_points.items():
        landmark = landmarks[landmark_idx]
        points[key] = np.array([landmark.x * w, landmark.y * h])
        if landmark.visibility < visibility_threshold and key in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            return {'visibility_issue': True, 'missing_points': key}
    
    # Calculate midpoints
    shoulder_mid = (points['left_shoulder'] + points['right_shoulder']) / 2
    hip_mid = (points['left_hip'] + points['right_hip']) / 2
    ear_mid = (points['left_ear'] + points['right_ear']) / 2
    eye_mid = (points['left_eye'] + points['right_eye']) / 2
    
    points['shoulder_mid'] = shoulder_mid
    points['hip_mid'] = hip_mid
    points['ear_mid'] = ear_mid
    points['eye_mid'] = eye_mid
    
    # Apply calibration if provided
    calibration_factor = 1.0
    if calibration_data:
        # Adjust measurements based on calibration reference
        calibration_factor = calibration_data.get('calibration_factor', 1.0)
    
    # 1. ENHANCED BACK ALIGNMENT ANALYSIS
    # Calculate the angle between hip, shoulder, and vertical
    back_angle = calculate_angle_with_vertical(hip_mid, shoulder_mid)
    
    # Calculate spine curvature (approximation using multiple points if available)
    spine_points = [hip_mid, shoulder_mid]
    if 'mid_back' in points:  # If we have a mid-back point
        spine_points.insert(1, points['mid_back'])
    
    # Calculate spine straightness score based on deviation from a straight line
    # Higher curvature = lower score
    back_curvature = 0
    if len(spine_points) > 2:
        for i in range(1, len(spine_points) - 1):
            # Calculate how much the middle point deviates from straight line
            pt_prev = spine_points[i-1]
            pt_curr = spine_points[i]
            pt_next = spine_points[i+1]
            angle = calculate_angle(pt_prev, pt_curr, pt_next)
            back_curvature += abs(180 - angle)
    
    # Adjusted thresholds based on calibration
    back_threshold1 = 10 * calibration_factor
    back_threshold2 = 20 * calibration_factor
    back_threshold3 = 30 * calibration_factor
    
    if 0 <= back_angle <= back_threshold1 and back_curvature < 10:
        back_status = "Excellent"
        back_feedback = "Your back posture is excellent"
        back_color = (0, 255, 0)  # Green
        back_score = 100
    elif 0 <= back_angle <= back_threshold2 and back_curvature < 20:
        back_status = "Good"
        back_feedback = "Your back is well positioned"
        back_color = (0, 255, 0)  # Green
        back_score = 80
    elif 0 <= back_angle <= back_threshold3 and back_curvature < 35:
        back_status = "Fair"
        back_feedback = "Straighten your back slightly"
        back_color = (0, 165, 255)  # Orange
        back_score = 60
    else:
        back_status = "Poor"
        back_feedback = "Sit up straight, you're leaning too far forward"
        back_color = (0, 0, 255)  # Red
        back_score = 30
    
    # 2. ENHANCED SHOULDER SYMMETRY ANALYSIS
    # Calculate multiple metrics for shoulder alignment
    
    # Height difference (normalized)
    shoulder_height_diff = (points['left_shoulder'][1] - points['right_shoulder'][1]) / h
    
    # Width difference (check if one shoulder is forward/backward)
    shoulder_width_ratio = abs(points['left_shoulder'][0] - shoulder_mid[0]) / abs(points['right_shoulder'][0] - shoulder_mid[0])
    
    # Calculate the angle of the shoulder line with horizontal
    shoulder_angle = abs(np.degrees(np.arctan2(
        points['left_shoulder'][1] - points['right_shoulder'][1], 
        points['left_shoulder'][0] - points['right_shoulder'][0])))
    
    # Calculate side lean angle
    side_lean = calculate_side_lean(
        points['left_shoulder'], points['right_shoulder'], 
        points['left_hip'], points['right_hip'])
    
    # Rotation check (facing the camera)
    rotation_factor, width_ratio = check_rotation(
        points['left_shoulder'], points['right_shoulder'],
        points['left_hip'], points['right_hip'], w)
    
    shoulder_score = 100
    
    # Penalize for height difference
    if abs(shoulder_height_diff) > 0.03:
        shoulder_score -= 30
    elif abs(shoulder_height_diff) > 0.015:
        shoulder_score -= 15
    
    # Penalize for width asymmetry (one shoulder forward)
    if shoulder_width_ratio < 0.7 or shoulder_width_ratio > 1.3:
        shoulder_score -= 25
    elif shoulder_width_ratio < 0.8 or shoulder_width_ratio > 1.2:
        shoulder_score -= 15
    
    # Penalize for side lean
    if side_lean > 10:
        shoulder_score -= 20
    elif side_lean > 5:
        shoulder_score -= 10
        
    # Penalize for rotation
    if rotation_factor > 0.2:
        shoulder_score -= 25
    elif rotation_factor > 0.1:
        shoulder_score -= 15
    
    # Clamp score to minimum of 0
    shoulder_score = max(0, shoulder_score)
    
    # Determine shoulder status based on score
    if shoulder_score >= 90:
        shoulder_status = "Excellent"
        shoulder_feedback = "Your shoulders are perfectly balanced"
        shoulder_color = (0, 255, 0)  # Green
    elif shoulder_score >= 70:
        shoulder_status = "Good"
        shoulder_feedback = "Your shoulders are well aligned"
        shoulder_color = (0, 255, 0)  # Green
    elif shoulder_score >= 40:
        shoulder_status = "Fair"
        if shoulder_height_diff > 0:
            shoulder_feedback = "Lower your left shoulder or raise your right shoulder"
        else:
            shoulder_feedback = "Lower your right shoulder or raise your left shoulder"
        shoulder_color = (0, 165, 255)  # Orange
    else:
        shoulder_status = "Poor"
        if rotation_factor > 0.15:
            shoulder_feedback = "Turn to face the camera directly"
        elif side_lean > 8:
            shoulder_feedback = "Avoid leaning to the side, balance your weight evenly"
        elif shoulder_height_diff > 0:
            shoulder_feedback = "Your shoulders are uneven. Lower your left shoulder"
        else:
            shoulder_feedback = "Your shoulders are uneven. Lower your right shoulder"
        shoulder_color = (0, 0, 255)  # Red
    
    # 3. ENHANCED NECK/HEAD POSITION ANALYSIS
    # Use multiple reference points for a more accurate assessment
    
    # Calculate head tilt (ear to ear line should be horizontal)
    head_tilt = abs(np.degrees(np.arctan2(
        points['left_ear'][1] - points['right_ear'][1], 
        points['left_ear'][0] - points['right_ear'][0])))
    
    # Look at ear position relative to shoulders (side view)
    ear_shoulder_horizontal_offset = (ear_mid[0] - shoulder_mid[0]) / w
    
    # Vertical alignment of head (ear should be aligned with shoulders vertically)
    ear_shoulder_vertical_offset = (ear_mid[1] - shoulder_mid[1]) / h
    
    # Calculate the angle between the vertical line from shoulders and the line to ears
    neck_angle = calculate_angle(
        [shoulder_mid[0], shoulder_mid[1] - 100],  # Point above shoulder (vertical)
        shoulder_mid,                               # Shoulder midpoint
        ear_mid                                     # Ear midpoint
    )
    
    # Adjust ideal neck angle based on camera position
    ideal_neck_angle = 75  # Typical value for a slightly elevated camera
    
    # Calculate a comprehensive neck score
    neck_score = 100
    
    # Penalize for forward head posture
    if ear_shoulder_horizontal_offset > 0.08:
        neck_score -= 40
    elif ear_shoulder_horizontal_offset > 0.06:
        neck_score -= 25
    elif ear_shoulder_horizontal_offset > 0.03:
        neck_score -= 10
    
    # Penalize for backward head posture
    if ear_shoulder_horizontal_offset < -0.06:
        neck_score -= 35
    elif ear_shoulder_horizontal_offset < -0.03:
        neck_score -= 15
    
    # Penalize for head tilt
    if head_tilt > 15:
        neck_score -= 25
    elif head_tilt > 8:
        neck_score -= 10
    
    # Penalize for neck angle deviation
    neck_angle_deviation = abs(neck_angle - ideal_neck_angle)
    if neck_angle_deviation > 20:
        neck_score -= 30
    elif neck_angle_deviation > 10:
        neck_score -= 15
    
    # Clamp score
    neck_score = max(0, neck_score)
    
    # Determine neck status based on score
    if neck_score >= 90:
        neck_status = "Excellent"
        neck_feedback = "Your head and neck position is excellent"
        neck_color = (0, 255, 0)  # Green
    elif neck_score >= 70:
        neck_status = "Good"
        neck_feedback = "Your neck position is good"
        neck_color = (0, 255, 0)  # Green
    elif neck_score >= 40:
        neck_status = "Fair"
        if ear_shoulder_horizontal_offset > 0.05:
            neck_feedback = "Bring your head back to reduce forward neck posture"
        elif ear_shoulder_horizontal_offset < -0.05:
            neck_feedback = "Your head is too far back, align it with your shoulders"
        elif head_tilt > 10:
            neck_feedback = "Keep your head level, avoid tilting to one side"
        else:
            neck_feedback = "Try to align your head better with your shoulders"
        neck_color = (0, 165, 255)  # Orange
    else:
        neck_status = "Poor"
        if ear_shoulder_horizontal_offset > 0.07:
            neck_feedback = "Avoid forward head posture, pull your chin back"
        elif ear_shoulder_horizontal_offset < -0.06:
            neck_feedback = "Your head is too far back, align it with your shoulders"
        elif head_tilt > 15:
            neck_feedback = "Your head is tilted, keep it level with the horizon"
        else:
            neck_feedback = "Your neck posture needs significant correction"
        neck_color = (0, 0, 255)  # Red
    
    # 4. NEW: SITTING DURATION ANALYSIS
    # This would track how long the person has been sitting and suggest breaks
    # (Just a placeholder - actual implementation would require time tracking)
    sitting_duration_score = 100  # Placeholder score
    
    # 5. COMPREHENSIVE OVERALL POSTURE SCORE
    # Calculate a weighted score based on all metrics with more nuance
    max_score = 100
    back_weight = 0.35
    shoulder_weight = 0.3
    neck_weight = 0.35
    
    overall_score = int(
        back_weight * back_score + 
        shoulder_weight * shoulder_score + 
        neck_weight * neck_score
    )
    
    if overall_score >= 90:
        overall_status = "Excellent"
        overall_color = (0, 255, 0)  # Green
    elif overall_score >= 70:
        overall_status = "Good"
        overall_color = (0, 255, 0)  # Green
    elif overall_score >= 50:
        overall_status = "Fair"
        overall_color = (0, 165, 255)  # Orange
    else:
        overall_status = "Poor"
        overall_color = (0, 0, 255)  # Red
    
    # Additional metrics for comprehensive analysis
    detailed_metrics = {
        'back_angle': back_angle,
        'back_curvature': back_curvature,
        'shoulder_height_diff': shoulder_height_diff,
        'shoulder_width_ratio': shoulder_width_ratio, 
        'shoulder_angle': shoulder_angle,
        'side_lean': side_lean,
        'rotation_factor': rotation_factor,
        'head_tilt': head_tilt,
        'ear_shoulder_horizontal': ear_shoulder_horizontal_offset,
        'ear_shoulder_vertical': ear_shoulder_vertical_offset,
        'neck_angle': neck_angle,
        'neck_angle_deviation': neck_angle_deviation
    }
    
    return {
        'back': {
            'status': back_status, 
            'feedback': back_feedback, 
            'color': back_color, 
            'angle': back_angle,
            'score': back_score
        },
        'shoulder': {
            'status': shoulder_status, 
            'feedback': shoulder_feedback, 
            'color': shoulder_color,
            'diff': shoulder_height_diff,
            'score': shoulder_score
        },
        'neck': {
            'status': neck_status, 
            'feedback': neck_feedback, 
            'color': neck_color,
            'angle': neck_angle,
            'score': neck_score
        },
        'overall': {
            'status': overall_status,
            'score': overall_score,
            'color': overall_color
        },
        'points': points,
        'detailed_metrics': detailed_metrics
    }

def draw_posture_visualization(image, feedback):
    """Enhanced visualization with more detailed guides on the image."""
    h, w, c = image.shape
    points = feedback['points']
    
    # Draw skeleton reference lines
    # 1. Draw vertical reference line
    cv2.line(image, 
             (int(points['hip_mid'][0]), int(points['hip_mid'][1])), 
             (int(points['hip_mid'][0]), 0), 
             (200, 200, 200), 1, cv2.LINE_AA)
    
    # 2. Draw back line (spine)
    cv2.line(image, 
             (int(points['shoulder_mid'][0]), int(points['shoulder_mid'][1])), 
             (int(points['hip_mid'][0]), int(points['hip_mid'][1])), 
             feedback['back']['color'], 3, cv2.LINE_AA)
    
    # 3. Draw shoulder line
    cv2.line(image, 
             (int(points['left_shoulder'][0]), int(points['left_shoulder'][1])), 
             (int(points['right_shoulder'][0]), int(points['right_shoulder'][1])), 
             feedback['shoulder']['color'], 3, cv2.LINE_AA)
    
    # 4. Draw neck line
    cv2.line(image, 
             (int(points['shoulder_mid'][0]), int(points['shoulder_mid'][1])), 
             (int(points['ear_mid'][0]), int(points['ear_mid'][1])), 
             feedback['neck']['color'], 3, cv2.LINE_AA)
    
    # 5. Draw head orientation line
    cv2.line(image, 
             (int(points['left_ear'][0]), int(points['left_ear'][1])), 
             (int(points['right_ear'][0]), int(points['right_ear'][1])), 
             feedback['neck']['color'], 2, cv2.LINE_AA)
    
    # 6. Draw vertical reference from shoulder midpoint
    cv2.line(image, 
             (int(points['shoulder_mid'][0]), 0), 
             (int(points['shoulder_mid'][0]), h), 
             (200, 200, 200), 1, cv2.LINE_AA)
    
    # Draw circles at key points for better visualization
    key_points = ['shoulder_mid', 'hip_mid', 'ear_mid']
    for point in key_points:
        cv2.circle(image, 
                  (int(points[point][0]), int(points[point][1])), 
                  4, (255, 255, 0), -1, cv2.LINE_AA)
    
    # Optional: Add angle annotations for educational feedback
    metrics = feedback['detailed_metrics']
    
    # Show back angle
    back_angle_pos = (int(points['hip_mid'][0]) + 15, int((points['hip_mid'][1] + points['shoulder_mid'][1]) / 2))
    cv2.putText(image, f"{metrics['back_angle']:.1f}°", back_angle_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['back']['color'], 1, cv2.LINE_AA)
    
    # Show neck angle
    neck_angle_pos = (int(points['ear_mid'][0]) + 10, int((points['shoulder_mid'][1] + points['ear_mid'][1]) / 2))
    cv2.putText(image, f"{metrics['neck_angle']:.1f}°", neck_angle_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['neck']['color'], 1, cv2.LINE_AA)
    
    return image

def create_feedback_panel(image, feedback):
    """Create a sleek, premium feedback panel on the image."""
    h, w, c = image.shape
    
    # Panel dimensions for a more compact design
    panel_height = 130
    
    # Create a new image with extra height for the panel
    expanded_image = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    
    # Copy the original image to the top portion
    expanded_image[0:h, 0:w] = image
    
    # Create premium dark panel background
    expanded_image[h:h+panel_height, 0:w] = (18, 18, 24)  # Dark background
    
    # Add subtle gradient for premium feel
    for y in range(h, h + 20):
        alpha = (y - h) / 20  # Gradient factor
        cv2.line(expanded_image, (0, y), (w, y), 
                (20 + int(alpha * 10), 20 + int(alpha * 10), 26 + int(alpha * 10)), 1)
    
    # Add posture score with clean font styling
    score_text = f"Posture Score: {feedback['overall']['score']}"
    
    # Add rounded rectangle behind the score
    cv2.rectangle(expanded_image, 
                 (15, h + 15), 
                 (235, h + 45), 
                 (30, 30, 40), 
                 -1, 
                 cv2.LINE_AA)
    cv2.rectangle(expanded_image, 
                 (15, h + 15), 
                 (235, h + 45), 
                 feedback['overall']['color'], 
                 1,
                 cv2.LINE_AA)
    
    # Add score text
    cv2.putText(expanded_image, score_text, (25, h + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, feedback['overall']['color'], 2, cv2.LINE_AA)
    
    # Draw bars for each posture element
    bar_y = h + 55
    bar_height = 18
    bar_spacing = 25
    bar_max_width = w - 240
    
    # Function to draw modern, minimal bars
    def draw_premium_bar(y_pos, label, score, color, status):
        # Draw label
        cv2.putText(expanded_image, label, (20, y_pos + 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Background bar
        cv2.rectangle(expanded_image, 
                     (90, y_pos), 
                     (90 + bar_max_width, y_pos + bar_height), 
                     (35, 35, 45), 
                     -1, 
                     cv2.LINE_AA)
        
        # Score bar
        bar_width = int((score / 100) * bar_max_width)
        cv2.rectangle(expanded_image, 
                     (90, y_pos), 
                     (90 + bar_width, y_pos + bar_height), 
                     color, 
                     -1, 
                     cv2.LINE_AA)
        
        # Add subtle highlight
        highlight_height = 3
        highlight_alpha = 0.3
        highlight = expanded_image[y_pos:y_pos+highlight_height, 90:90+bar_width].copy()
        cv2.rectangle(highlight, (0, 0), (bar_width, highlight_height), (255, 255, 255), -1)
        cv2.addWeighted(highlight, highlight_alpha, 
                        expanded_image[y_pos:y_pos+highlight_height, 90:90+bar_width], 
                        1 - highlight_alpha, 0, 
                        expanded_image[y_pos:y_pos+highlight_height, 90:90+bar_width])
        
        # Add status text
        cv2.putText(expanded_image, status, (100 + bar_max_width, y_pos + 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Draw all three bars
    draw_premium_bar(bar_y, "Back:", feedback['back']['score'], feedback['back']['color'], feedback['back']['status'])
    draw_premium_bar(bar_y + bar_spacing, "Shoulders:", feedback['shoulder']['score'], feedback['shoulder']['color'], feedback['shoulder']['status'])
    draw_premium_bar(bar_y + bar_spacing * 2, "Neck/Head:", feedback['neck']['score'], feedback['neck']['color'], feedback['neck']['status'])
    
    # Add minimal footer
    footer_y = h + panel_height - 12
    cv2.putText(expanded_image, "PostureCoach™ | Press 'q' to exit", 
                (w - 230, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 150), 1, cv2.LINE_AA)
    
    # Add feedback banner
    banner_height = 35
    
    # Semi-transparent dark background for banner
    banner_overlay = image.copy()
    cv2.rectangle(banner_overlay, (0, h - banner_height), (w, h), (15, 15, 25), -1)
    
    # Apply the banner with transparency
    alpha = 0.75
    cv2.addWeighted(banner_overlay[h-banner_height:h, 0:w], alpha, 
                   image[h-banner_height:h, 0:w], 1-alpha, 0, 
                   expanded_image[h-banner_height:h, 0:w])
    
    # Add border line
    border_color = feedback['current_feedback_color']
    border_thickness = 2
    cv2.line(expanded_image, (0, h - banner_height), (w, h - banner_height), border_color, border_thickness)
    
    # Add the current feedback text
    feedback_text = feedback['current_feedback_text']
    
    # Calculate text position to center it
    text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - (banner_height // 2) + 5
    
    # Add text
    cv2.putText(expanded_image, feedback_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback['current_feedback_color'], 2, cv2.LINE_AA)
    
    return expanded_image

class PostureCalibrationData:
    """Class to store calibration data and user-specific adjustments."""
    def __init__(self):
        self.calibrated = False
        self.reference_back_angle = None
        self.reference_shoulder_angle = None
        self.reference_neck_angle = None
        self.ideal_posture_landmarks = None
        self.calibration_factor = 1.0
        self.user_height = None
        self.camera_position = "front"  # front, side, angle
        
    def calibrate(self, feedback, landmarks, user_height=None):
        """Store reference values from a good posture position."""
        self.reference_back_angle = feedback['detailed_metrics']['back_angle']
        self.reference_shoulder_angle = feedback['detailed_metrics']['shoulder_angle']
        self.reference_neck_angle = feedback['detailed_metrics']['neck_angle']
        self.ideal_posture_landmarks = landmarks
        if user_height:
            self.user_height = user_height
        self.calibrated = True
        
    # Continuing from where the code was cut off
    def get_calibration_data(self):
        """Return calibration data as a dictionary."""
        return {
            'calibrated': self.calibrated,
            'reference_back_angle': self.reference_back_angle,
            'reference_shoulder_angle': self.reference_shoulder_angle,
            'reference_neck_angle': self.reference_neck_angle,
            'calibration_factor': self.calibration_factor,
            'user_height': self.user_height,
            'camera_position': self.camera_position
        }
        
    def adjust_calibration(self, factor):
        """Adjust calibration sensitivity."""
        self.calibration_factor = factor


class PostureCoach:
    """Main class to run posture detection and feedback."""
    def __init__(self):
        self.calibration = PostureCalibrationData()
        self.feedback_history = deque(maxlen=30)  # Store last 30 frames of feedback
        self.current_feedback = None
        self.feedback_cooldown = 0
        self.feedback_threshold = 5  # Number of consistent frames before giving feedback
        self.last_feedback_time = 0
        self.feedback_interval = 5  # Seconds between feedback
        
    def run(self, camera_index=0, width=640, height=480):
        """Main execution loop for posture detection."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Setup MediaPipe Pose
        with mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                model_complexity=1) as pose:
            
            # Calibration mode flag
            is_calibration_mode = False
            calibration_countdown = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture video")
                    break
                
                # Flip the image horizontally for a mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Process the image and detect pose
                results = pose.process(image)
                
                # Convert back to BGR for OpenCV
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    # Get landmarks
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get posture feedback
                    if self.calibration.calibrated:
                        calibration_data = self.calibration.get_calibration_data()
                    else:
                        calibration_data = None
                    
                    feedback = get_posture_feedback(landmarks, image.shape[:2], calibration_data)
                    
                    # Handle calibration mode
                    if is_calibration_mode:
                        if calibration_countdown > 0:
                            # Display countdown
                            cv2.putText(image, f"Calibrating in {calibration_countdown}...", 
                                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            calibration_countdown -= 1
                        else:
                            # Perform calibration
                            self.calibration.calibrate(feedback, landmarks)
                            is_calibration_mode = False
                            print("Calibration complete!")
                    
                    # If we have a visibility issue, skip drawing and feedback
                    if 'visibility_issue' in feedback and feedback['visibility_issue']:
                        # Display warning
                        warning_text = f"Cannot see full body: {feedback.get('missing_points', 'unknown')} not visible"
                        cv2.putText(image, warning_text, (20, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        # Draw pose visualization
                        image = draw_posture_visualization(image, feedback)
                        
                        # Update feedback history
                        self.update_feedback(feedback)
                        
                        # Add feedback banner and panel
                        feedback['current_feedback_text'] = self.current_feedback['text']
                        feedback['current_feedback_color'] = self.current_feedback['color']
                        image = create_feedback_panel(image, feedback)
                else:
                    # No pose detected
                    cv2.putText(image, "No pose detected", (20, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Display control instructions
                cv2.putText(image, "Press 'c' to calibrate, 'q' to quit", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Display the image
                cv2.imshow('PostureCoach', image)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    is_calibration_mode = True
                    calibration_countdown = 3  # 3 second countdown
                    print("Calibration mode activated. Sit in your best posture.")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def update_feedback(self, feedback):
        """Update feedback history and determine current feedback to give."""
        # Prioritize feedback based on which aspect needs the most improvement
        scores = {
            'back': feedback['back']['score'],
            'shoulder': feedback['shoulder']['score'],
            'neck': feedback['neck']['score']
        }
        
        # Find the worst posture aspect
        worst_aspect = min(scores, key=scores.get)
        
        # Only update if below a good threshold
        if scores[worst_aspect] < 70:
            self.feedback_history.append(worst_aspect)
            
            # Count occurrences of each aspect in recent history
            back_count = sum(1 for x in self.feedback_history if x == 'back')
            shoulder_count = sum(1 for x in self.feedback_history if x == 'shoulder')
            neck_count = sum(1 for x in self.feedback_history if x == 'neck')
            
            # Determine if we should give feedback (needs to be consistent for X frames)
            current_time = time.time()
            if current_time - self.last_feedback_time > self.feedback_interval:
                if back_count >= self.feedback_threshold:
                    self.current_feedback = {
                        'text': feedback['back']['feedback'],
                        'color': feedback['back']['color']
                    }
                    self.last_feedback_time = current_time
                elif shoulder_count >= self.feedback_threshold:
                    self.current_feedback = {
                        'text': feedback['shoulder']['feedback'],
                        'color': feedback['shoulder']['color']
                    }
                    self.last_feedback_time = current_time
                elif neck_count >= self.feedback_threshold:
                    self.current_feedback = {
                        'text': feedback['neck']['feedback'],
                        'color': feedback['neck']['color']
                    }
                    self.last_feedback_time = current_time
        else:
            # Good posture detected
            if not self.current_feedback or scores[worst_aspect] > 85:
                self.current_feedback = {
                    'text': "Great posture! Keep it up!",
                    'color': (0, 255, 0)  # Green
                }
        
        # Ensure we have a default feedback if none set yet
        if not self.current_feedback:
            self.current_feedback = {
                'text': "Analyzing your posture...",
                'color': (255, 255, 255)  # White
            }


def main():
    """Main function to run the PostureCoach application."""
    print("Starting PostureCoach...")
    print("Initializing camera and models...")
    
    coach = PostureCoach()
    
    print("Ready! Starting posture detection.")
    print("Controls: Press 'c' to calibrate your posture, 'q' to quit.")
    
    # Run the application
    coach.run()
    
    print("PostureCoach closed. Thank you for using!")


if __name__ == "__main__":
    main()
