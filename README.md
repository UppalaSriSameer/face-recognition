
# Face Recognition (Minimal)

Minimal pipeline using OpenCV LBPH for **dataset → train → recognize**.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Place `data/haarcascade_frontalface_default.xml` in the `data/` folder.

## 1) Create dataset
Capture faces for a person:
```bash
cd src
python create_dataset.py --name Alice --samples 120
python create_dataset.py --name Bob --samples 120
```
This creates images in `../dataset/Alice` and `../dataset/Bob`.

## 2) Train model
```bash
python create_classifier.py
```
Saves:
- `../models/face_recognizer.xml`
- `../models/labels.json`

## 3) Run recognition
```bash
python predict.py
```
Press `q` to quit. Unknown faces are shown in red; known faces in green.

### Notes
- LBPH confidence is a distance; we convert it to a rough percentage. Tweak `--conf` (default 75) if you see false positives.
- Ensure you installed **opencv-contrib-python** (not just `opencv-python`) for the `cv2.face` module.

the results can vary based on the amount of images you have trained the model with and the ideal number would be 300-500
