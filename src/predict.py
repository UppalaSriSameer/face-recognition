
# src/predict.py
import os
import cv2
import argparse
from Detector import load_face_cascade, load_recognizer, annotate_frame_with_recognition

def main():
    parser = argparse.ArgumentParser(description="Run real-time face recognition.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--cascade", default=os.path.join("..", "data", "haarcascade_frontalface_default.xml"),
                        help="Path to Haar cascade.")
    parser.add_argument("--model", default=os.path.join("..", "models", "face_recognizer.xml"),
                        help="Path to trained LBPH model.")
    parser.add_argument("--labels", default=os.path.join("..", "models", "labels.json"),
                        help="Path to labels json.")
    parser.add_argument("--conf", type=float, default=75.0,
                        help="Confidence threshold (0-100). Higher = stricter.")
    args = parser.parse_args()

    face_cascade = load_face_cascade(args.cascade)
    recognizer, id_to_name = load_recognizer(args.model, args.labels)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("[INFO] Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = annotate_frame_with_recognition(frame, face_cascade, recognizer, id_to_name, args.conf)
            cv2.imshow("Face Recognition (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
