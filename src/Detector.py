
# src/Detector.py
import json
import cv2
import os

DEFAULT_CASCADE = os.path.join(os.path.dirname(__file__), "..", "data", "haarcascade_frontalface_default.xml")

def load_face_cascade(cascade_path: str = DEFAULT_CASCADE) -> cv2.CascadeClassifier:
    cascade_path = os.path.abspath(cascade_path)
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade.")
    return face_cascade

def load_recognizer(model_path: str, labels_path: str):
    model_path = os.path.abspath(model_path)
    labels_path = os.path.abspath(labels_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Recognizer model not found: {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)  # {"name_to_id": {...}, "id_to_name": {...}}
    id_to_name = {int(k): v for k, v in labels["id_to_name"].items()}
    return recognizer, id_to_name

def detect_faces(gray_frame, face_cascade,
                 scaleFactor: float = 1.1, minNeighbors: int = 5, minSize=(60, 60)):
    # returns list of (x, y, w, h)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
    )
    return faces

def draw_label(img, text, x, y):
    cv2.rectangle(img, (x, y - 22), (x + 8 + 8 * len(text), y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def annotate_frame_with_recognition(frame, face_cascade, recognizer, id_to_name, confidence_threshold: float = 75.0):
    """Detect faces and annotate frame with predicted names.
       Lower confidence means better match; we convert to a % score for readability."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, face_cascade)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200))
        label_id, confidence = recognizer.predict(roi_gray)  # confidence: 0=best

        name = id_to_name.get(label_id, "Unknown")
        # Convert LBPH "distance" to a rough confidence %. Clamp to [0, 100].
        conf_pct = max(0.0, min(100.0, 100.0 - confidence))
        display_name = name if conf_pct >= confidence_threshold else "Unknown"

        color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        draw_label(frame, f"{display_name} ({conf_pct:.0f}%)", x, y)

    return frame
