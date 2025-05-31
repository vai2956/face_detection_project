import cv2
import mediapipe as mp
import sys

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_and_draw(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.65) as face_detection:
        results = face_detection.process(image_rgb)
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bbox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                bbox_pixels = (
                    int(bbox.xmin * w),
                    int(bbox.ymin * h),
                    int(bbox.width * w),
                    int(bbox.height * h)
                )
                
                # Draw detection box
                mp_drawing.draw_detection(image, detection)
                
                # Add "Face" label under the box
                cv2.putText(image, 'Face', 
                           (bbox_pixels[0], bbox_pixels[1] + bbox_pixels[3] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"Detected face with confidence: {detection.score[0]:.2f}")
            
            # Add total face count to the frame
            face_count = len(results.detections)
            cv2.putText(image, f'Faces detected: {face_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("No faces detected in the image.")
            cv2.putText(image, 'No faces detected', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

def main():
    # Get input argument
    if len(sys.argv) < 2:
        print("Usage: python face_detection.py <image_path|webcam>")
        sys.exit(1)
    source = sys.argv[1]

    if source.lower() == "webcam":
        cap = cv2.VideoCapture(0)
        print("Press 'q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break
            frame = detect_and_draw(frame)
            cv2.imshow("MediaPipe Face Detection", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(source)
        if image is None:
            print(f"Error: Could not read image from {source}")
            sys.exit(1)
        image = detect_and_draw(image)
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 