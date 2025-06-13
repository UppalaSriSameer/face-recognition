
# src/create_dataset.py
import os
import cv2
import argparse
from datetime import datetime
from Detector import load_face_cascade, detect_faces

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Create a face dataset for one person.")
    parser.add_argument("--name", required=True, help="Person's name (folder name in dataset).")
    parser.add_argument("--samples", type=int, default=100, help="Number of face images to capture.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--dataset_dir", default=os.path.join("..", "dataset"), help="Dataset root directory.")
    parser.add_argument("--cascade", default=os.path.join("..", "data", "haarcascade_frontalface_default.xml"),
                        help="Path to Haar cascade.")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_dir)
    person_dir = os.path.join(dataset_root, args.name)
    ensure_dir(person_dir)

    face_cascade = load_face_cascade(args.cascade)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    count = 0
    print(f"[INFO] Capturing faces for '{args.name}'. Press 'q' to stop early.")
    try:
        while count < args.samples:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, face_cascade)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(person_dir, f"{args.name}_{timestamp}.jpg")
                cv2.imwrite(filename, roi)
                count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, f"Collected: {count}/{args.samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Create Dataset (q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if count >= args.samples:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[DONE] Saved {count} images to: {person_dir}")

if __name__ == "__main__":
    main()
