from transformers import DeformableDetrForObjectDetection, DetrImageProcessor
from PIL import Image
import torch
import cv2
import numpy as np

# ================= PATHS =================
MODEL_PATH = "hf_detr_army_output"
IMAGE_PATH = "09.jpg"      # change image name
OUT_PATH = "army_detected.jpg"

# ================= SETTINGS =================
CONF_THRESH = 0.60
REAL_DRONE_WIDTH = 0.45
FOCAL_LENGTH = 900
ALERT_DISTANCE = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD MODEL =================
processor = DetrImageProcessor.from_pretrained(MODEL_PATH)

model = DeformableDetrForObjectDetection.from_pretrained(
    MODEL_PATH
).to(device)

model.eval()

print("Using device:", device)

# ================= LOAD IMAGE =================
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    raise FileNotFoundError(f"Cannot open image: {IMAGE_PATH}")

h, w = frame.shape[:2]

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

image = Image.fromarray(rgb)

# ================= INFERENCE =================
inputs = processor(
    images=image,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([[h, w]]).to(device)

results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=CONF_THRESH
)[0]

boxes = results["boxes"]
scores = results["scores"]
labels = results["labels"]

# ================= KEEP ONLY BEST BOX =================
if len(scores) > 0:

    best_idx = torch.argmax(scores)

    score = scores[best_idx].item()
    label = labels[best_idx].item()

    box = boxes[best_idx].detach().cpu().numpy().astype(int)

    x1, y1, x2, y2 = box

    class_name = model.config.id2label.get(label, str(label))

    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)

    bbox_size = max(bbox_w, bbox_h)

    # ================= DISTANCE =================
    distance_m = (
        REAL_DRONE_WIDTH * FOCAL_LENGTH
    ) / bbox_size

    # ================= COLOR =================
    if distance_m <= ALERT_DISTANCE:
        color = (0, 0, 255)
        status = "ALERT"

    else:
        color = (0, 255, 0)
        status = "TRACKING"

    # ================= DRAW =================
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cv2.putText(
        frame,
        f"{class_name}",
        (x1, max(30, y1 - 60)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Conf: {score:.2f}",
        (x1, max(60, y1 - 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Distance: {distance_m:.1f} m",
        (x1, max(90, y1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

else:

    cv2.putText(
        frame,
        "NO DRONE DETECTED",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

# ================= SAVE =================
cv2.imwrite(OUT_PATH, frame)

print("Saved:", OUT_PATH)