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

def calculate_side_lean(left_hip, right_hip, left_ankle, right_ankle):
    """Calculate if the person is leaning to one side."""
    # Calculate the angle between the line connecting hips and the line connecting ankles
    hips_vector = np.array([right_hip[0] - left_hip[0], right_hip[1] - left_hip[1]])
    ankles_vector = np.array([right_ankle[0] - left_ankle[0], right_ankle[1] - left_ankle[1]])
    
    # Normalize vectors
    if np.linalg.norm(hips_vector) < 1e-6 or np.linalg.norm(ankles_vector) < 1e-6:
        return 0.0
    
    hips_vector = hips_vector / np.linalg.norm(hips_vector)
    ankles_vector = ankles_vector / np.linalg.norm(ankles_vector)
    
    # Calculate the angle
    dot_product = np.dot(hips_vector, ankles_vector)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def check_rotation(left_shoulder, right_shoulder, left_hip, right_hip, image_width):
    """Detect if the person is rotated (not facing the camera directly)."""
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

def calculate_weight_distribution(left_ankle, right_ankle, left_hip, right_hip):
    """Calculate weight distribution based on ankle and hip positions."""
    # Get the vertical difference between left and right hips
    hip_height_diff = abs(left_hip[1] - right_hip[1])
    
    # Get the vertical difference between left and right ankles
    ankle_height_diff = abs(left_ankle[1] - right_ankle[1])
    
    # Calculate lateral distance between hip midpoint and ankle midpoint
    hip_mid = (left_hip + right_hip) / 2
    ankle_mid = (left_ankle + right_ankle) / 2
    lateral_shift = abs(hip_mid[0] - ankle_mid[0])
    
    return hip_height_diff, ankle_height_diff, lateral_shift

def get_standing_posture_feedback(landmarks, image_shape, calibration_data=None):
    """
    Enhanced standing posture analysis with comprehensive metrics.
    
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
        'nose': mp_pose.PoseLandmark.NOSE.value,
        'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
        'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        'left_knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
        'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE.value
    }
    
    points = {}
    
    # Check if critical landmarks are visible
    for key, landmark_idx in key_points.items():
        landmark = landmarks[landmark_idx]
        points[key] = np.array([landmark.x * w, landmark.y * h])
        if landmark.visibility < visibility_threshold and key in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']:
            return {'visibility_issue': True, 'missing_points': key}
    
    # Calculate midpoints
    shoulder_mid = (points['left_shoulder'] + points['right_shoulder']) / 2
    hip_mid = (points['left_hip'] + points['right_hip']) / 2
    ear_mid = (points['left_ear'] + points['right_ear']) / 2
    eye_mid = (points['left_eye'] + points['right_eye']) / 2
    ankle_mid = (points['left_ankle'] + points['right_ankle']) / 2
    knee_mid = (points['left_knee'] + points['right_knee']) / 2
    
    points['shoulder_mid'] = shoulder_mid
    points['hip_mid'] = hip_mid
    points['ear_mid'] = ear_mid
    points['eye_mid'] = eye_mid
    points['ankle_mid'] = ankle_mid
    points['knee_mid'] = knee_mid
    
    # Apply calibration if provided
    calibration_factor = 1.0
    if calibration_data:
        # Adjust measurements based on calibration reference
        calibration_factor = calibration_data.get('calibration_factor', 1.0)
    
    # 1. SPINE ALIGNMENT ANALYSIS
    # Calculate the angle between shoulders, hips, and ankles
    upper_spine_angle = calculate_angle_with_vertical(hip_mid, shoulder_mid)
    lower_spine_angle = calculate_angle_with_vertical(ankle_mid, hip_mid)
    
    # Check if spine is straight (shoulders directly above hips, hips above ankles)
    horizontal_shoulder_hip_offset = abs(shoulder_mid[0] - hip_mid[0]) / w
    horizontal_hip_ankle_offset = abs(hip_mid[0] - ankle_mid[0]) / w
    
    spine_alignment_score = 100
    
    # Penalize for leaning forward/backward
    if horizontal_shoulder_hip_offset > 0.05:
        spine_alignment_score -= 20
    elif horizontal_shoulder_hip_offset > 0.03:
        spine_alignment_score -= 10
    
    if horizontal_hip_ankle_offset > 0.05:
        spine_alignment_score -= 20
    elif horizontal_hip_ankle_offset > 0.03:
        spine_alignment_score -= 10
    
    # Penalize for upper spine angle deviation
    if upper_spine_angle > 10:
        spine_alignment_score -= 20
    elif upper_spine_angle > 5:
        spine_alignment_score -= 10
    
    # Penalize for lower spine angle deviation
    if lower_spine_angle > 10:
        spine_alignment_score -= 20
    elif lower_spine_angle > 5:
        spine_alignment_score -= 10
    
    # Clamp score
    spine_alignment_score = max(0, spine_alignment_score)
    
    # Determine spine alignment status based on score
    if spine_alignment_score >= 90:
        spine_status = "Excellent"
        spine_feedback = "Your spine alignment is excellent"
        spine_color = (0, 255, 0)  # Green
    elif spine_alignment_score >= 70:
        spine_status = "Good"
        spine_feedback = "Your spine is well aligned"
        spine_color = (0, 255, 0)  # Green
    elif spine_alignment_score >= 40:
        spine_status = "Fair"
        if horizontal_shoulder_hip_offset > horizontal_hip_ankle_offset:
            spine_feedback = "Align your shoulders directly above your hips"
        else:
            spine_feedback = "Center your weight more evenly above your ankles"
        spine_color = (0, 165, 255)  # Orange
    else:
        spine_status = "Poor"
        if upper_spine_angle > lower_spine_angle:
            spine_feedback = "Straighten your upper back, shoulders should be aligned with hips"
        else:
            spine_feedback = "Align your body vertically from ankles through hips to shoulders"
        spine_color = (0, 0, 255)  # Red
    
    # 2. WEIGHT DISTRIBUTION ANALYSIS
    hip_height_diff, ankle_height_diff, lateral_shift = calculate_weight_distribution(
        points['left_ankle'], points['right_ankle'], points['left_hip'], points['right_hip'])
    
    # Normalize by image height
    normalized_hip_diff = hip_height_diff / h
    normalized_ankle_diff = ankle_height_diff / h
    normalized_lateral_shift = lateral_shift / w
    
    # Calculate side lean
    side_lean = calculate_side_lean(
        points['left_hip'], points['right_hip'], 
        points['left_ankle'], points['right_ankle'])
    
    weight_distribution_score = 100
    
    # Penalize for uneven hips
    if normalized_hip_diff > 0.03:
        weight_distribution_score -= 30
    elif normalized_hip_diff > 0.015:
        weight_distribution_score -= 15
    
    # Penalize for uneven ankles
    if normalized_ankle_diff > 0.02:
        weight_distribution_score -= 20
    elif normalized_ankle_diff > 0.01:
        weight_distribution_score -= 10
    
    # Penalize for lateral shift
    if normalized_lateral_shift > 0.05:
        weight_distribution_score -= 30
    elif normalized_lateral_shift > 0.03:
        weight_distribution_score -= 15
    
    # Penalize for side lean
    if side_lean > 10:
        weight_distribution_score -= 25
    elif side_lean > 5:
        weight_distribution_score -= 10
    
    # Clamp score
    weight_distribution_score = max(0, weight_distribution_score)
    
    # Determine weight distribution status based on score
    if weight_distribution_score >= 90:
        weight_status = "Excellent"
        weight_feedback = "Your weight is perfectly distributed"
        weight_color = (0, 255, 0)  # Green
    elif weight_distribution_score >= 70:
        weight_status = "Good"
        weight_feedback = "Your weight distribution is balanced"
        weight_color = (0, 255, 0)  # Green
    elif weight_distribution_score >= 40:
        weight_status = "Fair"
        if normalized_hip_diff > normalized_ankle_diff:
            weight_feedback = "Level your hips to distribute weight more evenly"
        else:
            weight_feedback = "Distribute your weight evenly on both feet"
        weight_color = (0, 165, 255)  # Orange
    else:
        weight_status = "Poor"
        if side_lean > 8:
            weight_feedback = "You're leaning to one side. Distribute weight evenly on both feet"
        elif normalized_hip_diff > 0.025:
            weight_feedback = "Your hips are uneven. Level them for better balance"
        else:
            weight_feedback = "Shift your weight to center over both feet evenly"
        weight_color = (0, 0, 255)  # Red
    
    # 3. SHOULDER RELAXATION ANALYSIS
    # Calculate shoulder to ear distance (should be greater when relaxed)
    left_shoulder_ear_dist = np.linalg.norm(points['left_shoulder'] - points['left_ear']) / h
    right_shoulder_ear_dist = np.linalg.norm(points['right_shoulder'] - points['right_ear']) / h
    
    # Calculate shoulder elevation (height relative to neck)
    left_shoulder_elevation = (ear_mid[1] - points['left_shoulder'][1]) / h
    right_shoulder_elevation = (ear_mid[1] - points['right_shoulder'][1]) / h
    
    # Detect shoulder tension by comparing to expected values
    # (Lower values indicate shoulders raised closer to ears)
    shoulder_relaxation_score = 100
    
    # Penalize for raised shoulders (shorter distance)
    # Adjust thresholds based on body proportions
    left_threshold = 0.18 * calibration_factor
    right_threshold = 0.18 * calibration_factor
    
    if left_shoulder_ear_dist < left_threshold - 0.04:
        shoulder_relaxation_score -= 30
    elif left_shoulder_ear_dist < left_threshold - 0.02:
        shoulder_relaxation_score -= 15
    
    if right_shoulder_ear_dist < right_threshold - 0.04:
        shoulder_relaxation_score -= 30
    elif right_shoulder_ear_dist < right_threshold - 0.02:
        shoulder_relaxation_score -= 15
    
    # Penalize for shoulder asymmetry
    shoulder_diff = abs(left_shoulder_ear_dist - right_shoulder_ear_dist) / ((left_shoulder_ear_dist + right_shoulder_ear_dist) / 2)
    if shoulder_diff > 0.15:
        shoulder_relaxation_score -= 20
    elif shoulder_diff > 0.08:
        shoulder_relaxation_score -= 10
    
    # Clamp score
    shoulder_relaxation_score = max(0, shoulder_relaxation_score)
    
    # Determine shoulder status based on score
    if shoulder_relaxation_score >= 90:
        shoulder_status = "Excellent"
        shoulder_feedback = "Your shoulders are perfectly relaxed"
        shoulder_color = (0, 255, 0)  # Green
    elif shoulder_relaxation_score >= 70:
        shoulder_status = "Good"
        shoulder_feedback = "Your shoulders are well relaxed"
        shoulder_color = (0, 255, 0)  # Green
    elif shoulder_relaxation_score >= 40:
        shoulder_status = "Fair"
        if left_shoulder_ear_dist < right_shoulder_ear_dist:
            shoulder_feedback = "Relax your left shoulder down away from your ear"
        else:
            shoulder_feedback = "Relax your right shoulder down away from your ear"
        shoulder_color = (0, 165, 255)  # Orange
    else:
        shoulder_status = "Poor"
        if shoulder_diff > 0.12:
            shoulder_feedback = "Your shoulders are uneven. Relax them both down evenly"
        elif left_shoulder_ear_dist < right_shoulder_ear_dist:
            shoulder_feedback = "Your shoulders are tense. Drop your left shoulder down"
        else:
            shoulder_feedback = "Your shoulders are tense. Drop your right shoulder down"
        shoulder_color = (0, 0, 255)  # Red
    
    # 4. COMPREHENSIVE OVERALL POSTURE SCORE
    # Calculate a weighted score based on all metrics
    max_score = 100
    spine_weight = 0.35
    weight_weight = 0.35
    shoulder_weight = 0.30
    
    overall_score = int(
        spine_weight * spine_alignment_score + 
        weight_weight * weight_distribution_score + 
        shoulder_weight * shoulder_relaxation_score
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
        'upper_spine_angle': upper_spine_angle,
        'lower_spine_angle': lower_spine_angle,
        'horizontal_shoulder_hip_offset': horizontal_shoulder_hip_offset,
        'horizontal_hip_ankle_offset': horizontal_hip_ankle_offset,
        'hip_height_diff': normalized_hip_diff,
        'ankle_height_diff': normalized_ankle_diff,
        'lateral_shift': normalized_lateral_shift,
        'side_lean': side_lean,
        'left_shoulder_ear_dist': left_shoulder_ear_dist,
        'right_shoulder_ear_dist': right_shoulder_ear_dist,
        'shoulder_diff': shoulder_diff
    }
    
    return {
        'spine': {
            'status': spine_status, 
            'feedback': spine_feedback, 
            'color': spine_color,
            'score': spine_alignment_score
        },
        'weight': {
            'status': weight_status, 
            'feedback': weight_feedback, 
            'color': weight_color,
            'score': weight_distribution_score
        },
        'shoulder': {
            'status': shoulder_status, 
            'feedback': shoulder_feedback, 
            'color': shoulder_color,
            'score': shoulder_relaxation_score
        },
        'overall': {
            'status': overall_status,
            'score': overall_score,
            'color': overall_color
        },
        'points': points,
        'detailed_metrics': detailed_metrics
    }

def draw_standing_posture_visualization(image, feedback):
    """Enhanced visualization with detailed guides on the image for standing posture."""
    h, w, c = image.shape
    points = feedback['points']
    
    # Draw skeleton reference lines
    # 1. Draw vertical reference line from ankle mid to top
    cv2.line(image, 
             (int(points['ankle_mid'][0]), int(points['ankle_mid'][1])), 
             (int(points['ankle_mid'][0]), 0), 
             (200, 200, 200), 1, cv2.LINE_AA)
    
    # 2. Draw lower spine line (ankles to hips)
    cv2.line(image, 
             (int(points['ankle_mid'][0]), int(points['ankle_mid'][1])), 
             (int(points['hip_mid'][0]), int(points['hip_mid'][1])), 
             feedback['spine']['color'], 3, cv2.LINE_AA)
    
    # 3. Draw upper spine line (hips to shoulders)
    cv2.line(image, 
             (int(points['hip_mid'][0]), int(points['hip_mid'][1])), 
             (int(points['shoulder_mid'][0]), int(points['shoulder_mid'][1])), 
             feedback['spine']['color'], 3, cv2.LINE_AA)
    
    # 4. Draw hip line
    cv2.line(image, 
             (int(points['left_hip'][0]), int(points['left_hip'][1])), 
             (int(points['right_hip'][0]), int(points['right_hip'][1])), 
             feedback['weight']['color'], 3, cv2.LINE_AA)
    
    # 5. Draw ankle line
    cv2.line(image, 
             (int(points['left_ankle'][0]), int(points['left_ankle'][1])), 
             (int(points['right_ankle'][0]), int(points['right_ankle'][1])), 
             feedback['weight']['color'], 3, cv2.LINE_AA)
    
    # 6. Draw shoulder line
    cv2.line(image, 
             (int(points['left_shoulder'][0]), int(points['left_shoulder'][1])), 
             (int(points['right_shoulder'][0]), int(points['right_shoulder'][1])), 
             feedback['shoulder']['color'], 3, cv2.LINE_AA)
    
    # 7. Draw neck line
    cv2.line(image, 
             (int(points['shoulder_mid'][0]), int(points['shoulder_mid'][1])), 
             (int(points['ear_mid'][0]), int(points['ear_mid'][1])), 
             feedback['shoulder']['color'], 3, cv2.LINE_AA)
    
    # Draw circles at key points for better visualization
    key_points = ['shoulder_mid', 'hip_mid', 'ear_mid', 'ankle_mid']
    for point in key_points:
        cv2.circle(image, 
                  (int(points[point][0]), int(points[point][1])), 
                  4, (255, 255, 0), -1, cv2.LINE_AA)
    
    # Add angle annotations
    metrics = feedback['detailed_metrics']
    
    # Show upper spine angle
    upper_spine_angle_pos = (int(points['hip_mid'][0]) + 15, int((points['hip_mid'][1] + points['shoulder_mid'][1]) / 2))
    cv2.putText(image, f"{metrics['upper_spine_angle']:.1f}°", upper_spine_angle_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['spine']['color'], 1, cv2.LINE_AA)
    
    # Show lower spine angle
    lower_spine_angle_pos = (int(points['ankle_mid'][0]) + 15, int((points['ankle_mid'][1] + points['hip_mid'][1]) / 2))
    cv2.putText(image, f"{metrics['lower_spine_angle']:.1f}°", lower_spine_angle_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['spine']['color'], 1, cv2.LINE_AA)
    
    # Show hip height difference
    hip_diff_pos = (int((points['left_hip'][0] + points['right_hip'][0]) / 2), int(points['hip_mid'][1]) - 15)
    cv2.putText(image, f"Hip diff: {metrics['hip_height_diff']*100:.1f}%", hip_diff_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback['weight']['color'], 1, cv2.LINE_AA)
    
    return image

def create_standing_feedback_panel(frame, feedback_data):
    """Create a more compact feedback panel for standing posture."""
    # Get original frame dimensions
    h, w, c = frame.shape
    
    # Create a smaller panel (similar to sitting frame size)
    panel_height = int(h * 0.8)  # Make it 80% of original height
    expanded_image = frame.copy()
    expanded_image = cv2.resize(expanded_image, (int(panel_height * w/h), panel_height))
    
    # Add black bar at the bottom for feedback text (fixed height)
    footer_height = 120
    footer = np.zeros((footer_height, expanded_image.shape[1], 3), dtype=np.uint8)
    
    # Combine image with footer
    expanded_image = np.vstack((expanded_image, footer))
    
    # Get current feedback text and color
    feedback_text = feedback_data.get('current_feedback_text', 'Maintain good posture')
    feedback_color = feedback_data.get('current_feedback_color', (0, 255, 0))
    
    # Add feedback text to the footer
    text_x = 20
    text_y = expanded_image.shape[0] - 70
    
    # Add shadow for better readability
    cv2.putText(expanded_image, feedback_text, (text_x+1, text_y+1), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(expanded_image, feedback_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)
    
    # Add score visualization
    score = 0
    # Calculate total score (example: average of individual scores)
    if 'spine' in feedback_data and 'score' in feedback_data['spine']:
        score += feedback_data['spine']['score']
    if 'shoulder' in feedback_data and 'score' in feedback_data['shoulder']:
        score += feedback_data['shoulder']['score']
    if 'weight' in feedback_data and 'score' in feedback_data['weight']:
        score += feedback_data['weight']['score']
    
    score = int(score / 3)  # Average score
    
    # Score text
    score_text = f"Posture Score: {score}"
    score_x = 20
    score_y = expanded_image.shape[0] - 30
    
    # Score color (green if good, yellow if okay, red if poor)
    if score >= 80:
        score_color = (0, 255, 0)  # Green
    elif score >= 60:
        score_color = (0, 255, 255)  # Yellow
    else:
        score_color = (0, 0, 255)  # Red
    
    # Draw score text with shadow
    cv2.putText(expanded_image, score_text, (score_x+1, score_y+1), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(expanded_image, score_text, (score_x, score_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2, cv2.LINE_AA)
    
    # Add visual bar charts for different posture aspects
    bar_start_x = expanded_image.shape[1] // 2
    bar_width = expanded_image.shape[1] // 3
    bar_height = 15
    
    # Back posture
    back_y = score_y - 60
    cv2.putText(expanded_image, "Back:", (bar_start_x - 100, back_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw empty bar
    cv2.rectangle(expanded_image, (bar_start_x, back_y - bar_height), 
                 (bar_start_x + bar_width, back_y), (50, 50, 50), -1)
    
    # Fill with spine score
    spine_score = feedback_data.get('spine', {}).get('score', 70)
    spine_width = int(bar_width * spine_score / 100)
    spine_color = feedback_data.get('spine', {}).get('color', (0, 255, 0))
    cv2.rectangle(expanded_image, (bar_start_x, back_y - bar_height), 
                 (bar_start_x + spine_width, back_y), spine_color, -1)
    
    # Add "Excellent" text
    cv2.putText(expanded_image, "Excellent", (bar_start_x + bar_width + 10, back_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Shoulder posture
    shoulder_y = back_y - 25
    cv2.putText(expanded_image, "Shoulder:", (bar_start_x - 100, shoulder_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw empty bar
    cv2.rectangle(expanded_image, (bar_start_x, shoulder_y - bar_height), 
                 (bar_start_x + bar_width, shoulder_y), (50, 50, 50), -1)
    
    # Fill with shoulder score
    shoulder_score = feedback_data.get('shoulder', {}).get('score', 70)
    shoulder_width = int(bar_width * shoulder_score / 100)
    shoulder_color = feedback_data.get('shoulder', {}).get('color', (0, 255, 0))
    cv2.rectangle(expanded_image, (bar_start_x, shoulder_y - bar_height), 
                 (bar_start_x + shoulder_width, shoulder_y), shoulder_color, -1)
    
    # Add "Poor" text
    cv2.putText(expanded_image, "Poor", (bar_start_x + bar_width + 10, shoulder_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Neck posture
    neck_y = shoulder_y - 25
    cv2.putText(expanded_image, "Neck/Head:", (bar_start_x - 100, neck_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw empty bar
    cv2.rectangle(expanded_image, (bar_start_x, neck_y - bar_height), 
                 (bar_start_x + bar_width, neck_y), (50, 50, 50), -1)
    
    # Fill with weight score (using as neck score for this example)
    weight_score = feedback_data.get('weight', {}).get('score', 70)
    weight_width = int(bar_width * weight_score / 100)
    weight_color = feedback_data.get('weight', {}).get('color', (0, 255, 0))
    cv2.rectangle(expanded_image, (bar_start_x, neck_y - bar_height), 
                 (bar_start_x + weight_width, neck_y), weight_color, -1)
    
    # Add "Good" text
    cv2.putText(expanded_image, "Good", (bar_start_x + bar_width + 10, neck_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Add press 'q' to quit instruction
    cv2.putText(expanded_image, "Press 'q' to quit", 
               (expanded_image.shape[1] - 150, expanded_image.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    
    return expanded_image

class StandingPostureAnalyzer:
    """Class to analyze and provide feedback on standing posture."""
    
    def __init__(self):
        """Initialize the posture analyzer with required components."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Feedback history for smoothing
        self.spine_feedback_queue = deque(maxlen=10)
        self.weight_feedback_queue = deque(maxlen=10)
        self.shoulder_feedback_queue = deque(maxlen=10)
        
        # Calibration data (can be set during calibration phase)
        self.calibration_data = None
        
        # Current feedback state
        self.current_feedback_area = "spine"  # Cycle between spine, weight, shoulder
        self.feedback_switch_time = time.time()
        self.feedback_switch_interval = 5  # seconds between feedback focus areas
    
    def process_frame(self, frame):
        """Process a video frame and return analyzed frame with feedback visualizations."""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(frame_rgb)
        
        # Check if pose was detected
        if not results.pose_landmarks:
            # If no pose detected, display instruction
            h, w, c = frame.shape
            cv2.putText(frame, "No pose detected. Please stand in frame.", 
                       (w//6, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame
        
        # Get posture feedback
        posture_feedback = get_standing_posture_feedback(
            results.pose_landmarks.landmark, 
            frame.shape, 
            self.calibration_data)
        
        # Check for visibility issues
        if posture_feedback.get('visibility_issue'):
            missing = posture_feedback.get('missing_points', '')
            cv2.putText(frame, f"Cannot see full body. Missing {missing}.", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame
        
        # Update feedback queues
        self.spine_feedback_queue.append(posture_feedback['spine']['feedback'])
        self.weight_feedback_queue.append(posture_feedback['weight']['feedback'])
        self.shoulder_feedback_queue.append(posture_feedback['shoulder']['feedback'])
        
        # Decide which feedback to show (cycling through areas)
        current_time = time.time()
        if current_time - self.feedback_switch_time > self.feedback_switch_interval:
            # Switch to next feedback area
            if self.current_feedback_area == "spine":
                self.current_feedback_area = "weight"
            elif self.current_feedback_area == "weight":
                self.current_feedback_area = "shoulder"
            else:
                self.current_feedback_area = "spine"
            
            self.feedback_switch_time = current_time
        
        # Choose current feedback to display based on area
        if self.current_feedback_area == "spine":
            current_feedback = self.get_most_common_feedback(self.spine_feedback_queue)
            current_feedback_color = posture_feedback['spine']['color']
        elif self.current_feedback_area == "weight":
            current_feedback = self.get_most_common_feedback(self.weight_feedback_queue)
            current_feedback_color = posture_feedback['weight']['color']
        else:  # shoulder
            current_feedback = self.get_most_common_feedback(self.shoulder_feedback_queue)
            current_feedback_color = posture_feedback['shoulder']['color']
        
        # Add current feedback to the feedback dict
        posture_feedback['current_feedback_text'] = current_feedback
        posture_feedback['current_feedback_color'] = current_feedback_color
        
        # Draw visualization guides on the image
        frame_with_guides = draw_standing_posture_visualization(frame, posture_feedback)
        
        # Create feedback panel and return
        return create_standing_feedback_panel(frame_with_guides, posture_feedback)
    
    def get_most_common_feedback(self, feedback_queue):
        """Get the most common feedback from the queue to avoid rapid feedback changes."""
        if not feedback_queue:
            return "Stand straight with shoulders relaxed"
            
        # Count occurrences of each feedback
        feedback_counts = {}
        for feedback in feedback_queue:
            if feedback in feedback_counts:
                feedback_counts[feedback] += 1
            else:
                feedback_counts[feedback] = 1
        
        # Return the most common feedback
        return max(feedback_counts.items(), key=lambda x: x[1])[0]
    
    def set_calibration_data(self, calibration_data):
        """Set calibration data from a reference posture."""
        self.calibration_data = calibration_data
    
    def release(self):
        """Release resources."""
        self.pose.close()

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set resolution (can adjust based on performance needs)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize analyzer
    analyzer = StandingPostureAnalyzer()
    
    # Initialize calibration state
    calibration_mode = False
    calibration_countdown = 0
    calibration_time = 3  # seconds
    last_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Handle calibration mode
        if calibration_mode:
            current_time = time.time()
            elapsed = current_time - last_time
            remaining = calibration_time - elapsed
            
            if remaining > 0:
                # Display countdown
                h, w, c = frame.shape
                cv2.putText(frame, f"Calibrating in {remaining:.1f}s...", 
                           (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.putText(frame, "Stand in your best posture", 
                           (w//4, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Perform calibration
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = analyzer.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Get reference measurements from good posture
                    landmarks = results.pose_landmarks.landmark
                    
                    # Example: Use shoulder-ear distance as calibration reference
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                    
                    h, w, c = frame.shape
                    left_shoulder_point = np.array([left_shoulder.x * w, left_shoulder.y * h])
                    right_shoulder_point = np.array([right_shoulder.x * w, right_shoulder.y * h])
                    left_ear_point = np.array([left_ear.x * w, left_ear.y * h])
                    right_ear_point = np.array([right_ear.x * w, right_ear.y * h])
                    
                    left_shoulder_ear_dist = np.linalg.norm(left_shoulder_point - left_ear_point) / h
                    right_shoulder_ear_dist = np.linalg.norm(right_shoulder_point - right_ear_point) / h
                    
                    calibration_data = {
                        'left_shoulder_ear_reference': left_shoulder_ear_dist,
                        'right_shoulder_ear_reference': right_shoulder_ear_dist,
                        'calibration_factor': 1.0  # Could be adjusted based on body proportions
                    }
                    
                    analyzer.set_calibration_data(calibration_data)
                    
                    cv2.putText(frame, "Calibration complete!", 
                               (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Calibration failed - no pose detected", 
                               (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Reset calibration mode
                calibration_mode = False
        else:
            # Normal mode - analyze posture
            frame = analyzer.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Standing Posture Analysis', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Enter calibration mode
            calibration_mode = True
            last_time = time.time()
    
    # Release resources
    cap.release()
    analyzer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
