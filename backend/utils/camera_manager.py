import cv2
import logging

class CameraManager:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger('CameraManager')
        self.logger.setLevel(logging.INFO)

    def __enter__(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera at index {self.camera_index}")
                
            self.logger.info(f"Camera {self.camera_index} initialized")
            return self
            
        except Exception as e:
            self.logger.error(f"Camera init failed: {str(e)}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera released")

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not initialized")
            
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
            
        return cv2.flip(frame, 1)  # Mirror effect

    def is_camera_available(self):
        temp_cap = cv2.VideoCapture(self.camera_index)
        is_available = temp_cap.isOpened()
        temp_cap.release()
        return is_available