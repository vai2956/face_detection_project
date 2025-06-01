import cv2
import mediapipe as mp
import sys
from typing import List, Tuple, Dict
import numpy as np

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Define facial feature landmarks
FACIAL_FEATURES: Dict[str, List[int]] = {
    "nose": [1],
    "forehead": [10],
    "lips": [13, 14],
    "eyes": [33, 133, 362, 263],
    "eyebrows": [70, 300]
}

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.65):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6
        )

    def process_frame(self, image: np.ndarray, mode: str, features_to_detect: List[str] = None) -> np.ndarray:
        """Process a single frame for face detection and/or feature detection."""
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Process face detection
        if mode == "detect" or mode == "both":
            detection_results = self.face_detection.process(image_rgb)
            if detection_results.detections:
                for detection in detection_results.detections:
                    mp_drawing.draw_detection(image, detection)
                    bbox = detection.location_data.relative_bounding_box
                    bbox_pixels = (
                        int(bbox.xmin * w),
                        int(bbox.ymin * h),
                        int(bbox.width * w),
                        int(bbox.height * h)
                    )
                    cv2.putText(image, 'Face', 
                               (bbox_pixels[0], bbox_pixels[1] + bbox_pixels[3] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                face_count = len(detection_results.detections)
                cv2.putText(image, f'Faces detected: {face_count}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process facial features
        if mode == "features" or mode == "both":
            mesh_results = self.face_mesh.process(image_rgb)
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    self._draw_features(image, landmarks, features_to_detect, w, h)
            else:
                cv2.putText(image, 'No facial features detected', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image

    def _draw_features(self, image: np.ndarray, landmarks: List, features: List[str], w: int, h: int) -> None:
        """Draw detected facial features on the image."""
        colors = {
            "nose": (0, 255, 255),
            "forehead": (255, 0, 255),
            "lips": (0, 0, 255),
            "eyes": (255, 255, 0),
            "eyebrows": (0, 255, 0)
        }

        for feature in features:
            if feature in FACIAL_FEATURES:
                indices = FACIAL_FEATURES[feature]
                points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
                
                # Draw points and connect them
                for point in points:
                    cv2.circle(image, point, 3, colors[feature], -1)
                
                # Draw feature name
                if points:
                    cv2.putText(image, feature.capitalize(), 
                               (points[0][0], points[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[feature], 1)

def main():
    if len(sys.argv) < 3:
        print("Usage: python face_detection.py <detect|features|both> <image_path|webcam> [features_to_detect]")
        print("Example: python face_detection.py both webcam nose lips eyes")
        sys.exit(1)

    try:
        mode = sys.argv[1].lower()
        if mode not in ["detect", "features", "both"]:
            raise ValueError("Invalid mode. Use 'detect', 'features', or 'both'")

        source = sys.argv[2].lower()
        features_to_detect = [arg.lower() for arg in sys.argv[3:]] if len(sys.argv) > 3 else []

        detector = FaceDetector()

        if source == "webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open webcam")

            print("Press 'q' to quit.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue

                processed_frame = detector.process_frame(frame, mode, features_to_detect)
                if processed_frame is not None:
                    cv2.imshow("Face Detection", processed_frame)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            image = cv2.imread(source)
            if image is None:
                raise RuntimeError(f"Could not read image from {source}")

            processed_image = detector.process_frame(image, mode, features_to_detect)
            if processed_image is not None:
                cv2.imshow("Result", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 