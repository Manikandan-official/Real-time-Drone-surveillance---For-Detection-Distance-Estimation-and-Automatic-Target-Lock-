from transformers import DeformableDetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw
import torch
import cv2
import numpy as np

# ---------------- PATHS ----------------
MODEL_PATH = "hf_detr_army_output"
VIDEO_PATH = "shaded.mp4"
OUT_PATH = "/teamspace/studios/this_studio/drone_alert_output3.mp4"

# ---------------- DISTANCE SETTINGS ----------------
REAL_DRONE_WIDTH = 0.45   # meters
FOCAL_LENGTH = 900        # tune experimentally

# ALERT distance threshold
ALERT_DISTANCE = 3       # meters

# ---------------- LOAD MODEL ----------------
model = DeformableDetrForObjectDetection.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    ignore_mismatched_sizes=True
)

processor = DetrImageProcessor.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

print("Using device:", device)

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUT_PATH,
    fourcc,
    fps,
    (w, h)
)

frame_id = 0

# ---------------- INFERENCE LOOP ----------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits.sigmoid()[0, :, 0].cpu()
    boxes = outputs.pred_boxes[0].cpu()

    keep = scores > 0.3

    draw = ImageDraw.Draw(image)

    alert_triggered = False

    # ---------------- DETECTIONS ----------------
    for score, box in zip(scores[keep], boxes[keep]):

        cx, cy, bw, bh = box.tolist()

        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        bbox_size = max(bbox_w, bbox_h)

        # Distance estimation
        if bbox_size > 0:
            distance_m = (REAL_DRONE_WIDTH * FOCAL_LENGTH) / bbox_size
        else:
            distance_m = 999

        # ---------------- ALERT CONDITION ----------------
        if distance_m <= ALERT_DISTANCE:

            alert_triggered = True

            box_color = "red"

        else:

            box_color = "lime"

        # ---------------- DRAW BOX ----------------
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=box_color,
            width=4
        )

        # ONLY DISTANCE ON TOP
        draw.text(
            (x1, y1 - 25),
            f"{distance_m:.1f} m",
            fill=box_color
        )

    # ---------------- GLOBAL ALERT ----------------
    if alert_triggered:

        draw.rectangle(
            [0, 0, w, 70],
            fill="red"
        )

        draw.text(
            (20, 20),
            "⚠ ALERT: DRONE TOO CLOSE ⚠",
            fill="white"
        )

    # ---------------- SAVE FRAME ----------------
    out_frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    out.write(out_frame)

    frame_id += 1

    if frame_id % 20 == 0:
        print("Processed frames:", frame_id)

# ---------------- CLEANUP ----------------
cap.release()
out.release()

print("Saved video:", OUT_PATH)