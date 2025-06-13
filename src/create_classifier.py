
# src/create_classifier.py
import os
import cv2
import json
import argparse
import numpy as np

def load_images_and_labels(dataset_root: str):
    X, y = [], []
    name_to_id = {}
    next_id = 0

    for person in sorted(os.listdir(dataset_root)):
        person_dir = os.path.join(dataset_root, person)
        if not os.path.isdir(person_dir):
            continue

        if person not in name_to_id:
            name_to_id[person] = next_id
            next_id += 1

        label_id = name_to_id[person]
        for file in os.listdir(person_dir):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(person_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # normalize size for recognizer
            img = cv2.resize(img, (200, 200))
            X.append(img)
            y.append(label_id)

    if not X:
        raise RuntimeError(f"No images found in dataset: {dataset_root}")

    id_to_name = {v: k for k, v in name_to_id.items()}
    return X, y, {"name_to_id": name_to_id, "id_to_name": id_to_name}

def main():
    parser = argparse.ArgumentParser(description="Train LBPH face recognizer from dataset.")
    parser.add_argument("--dataset_dir", default=os.path.join("..", "dataset"), help="Dataset root directory.")
    parser.add_argument("--models_dir", default=os.path.join("..", "models"), help="Where to save the model.")
    parser.add_argument("--model_name", default="face_recognizer.xml", help="Model filename.")
    parser.add_argument("--labels_name", default="labels.json", help="Labels filename.")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_dir)
    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    X, y, labels = load_images_and_labels(dataset_root)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8  # sensible defaults
    )
    import numpy as _np
    recognizer.train(X, _np.array(y))

    model_path = os.path.join(models_dir, args.model_name)
    labels_path = os.path.join(models_dir, args.labels_name)

    recognizer.write(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"[DONE] Saved model:  {model_path}")
    print(f"[DONE] Saved labels: {labels_path}")

if __name__ == "__main__":
    main()
