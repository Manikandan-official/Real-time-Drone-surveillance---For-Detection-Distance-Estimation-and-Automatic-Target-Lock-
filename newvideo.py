from transformers import DeformableDetrForObjectDetection, DetrImageProcessor
from PIL import Image
import torch
import cv2
import numpy as np

# ================= PATHS =================
MODEL_PATH = "hf_detr_army_output"
VIDEO_PATH = "shaded136.mp4"
OUT_PATH = "/teamspace/studios/this_studio/drone_alert_output_fixed.mp4"

# ================= SETTINGS =================
CONF_THRESH = 0.60          # increase if too many boxes
REAL_DRONE_WIDTH = 0.45
FOCAL_LENGTH = 900
ALERT_DISTANCE = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DeformableDetrForObjectDetection.from_pretrained(MODEL_PATH).to(device)
model.eval()

print("Using device:", device)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt").to(device)

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

        distance_m = (REAL_DRONE_WIDTH * FOCAL_LENGTH) / bbox_size

        if distance_m <= ALERT_DISTANCE:
            color = (0, 0, 255)
            status = "ALERT: DRONE TOO CLOSE"
        else:
            color = (0, 255, 0)
            status = "DRONE TRACKING"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        cv2.putText(
            frame,
            f"{class_name} {score:.2f}",
            (x1, max(30, y1 - 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Distance: {distance_m:.1f} m",
            (x1, max(60, y1 - 10)),
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

    out.write(frame)

    frame_id += 1
    if frame_id % 20 == 0:
        print("Processed:", frame_id)

cap.release()
out.release()

print("Saved:", OUT_PATH)