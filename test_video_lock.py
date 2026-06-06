import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DeformableDetrForObjectDetection, DetrImageProcessor

VIDEO_PATH = "test1.mp4"
MODEL_PATH = "./hf_detr_large_output_v2"   # change if needed
OUTPUT_PATH = "test1_output_locked.mp4"

CONF_THRESH = 0.25
REAL_DRONE_WIDTH_M = 0.45
FOCAL_LENGTH_PX = 900
LOCK_RADIUS = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DeformableDetrForObjectDetection.from_pretrained(MODEL_PATH).to(device)
model.eval()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0:
    fps = 25

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

cx, cy = w // 2, h // 2

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    boxes = outputs.pred_boxes[0]

    probs = logits.softmax(-1)

    # Remove no-object class
    scores, labels = probs[:, :-1].max(-1)

    keep = scores > CONF_THRESH

    best_box = None
    best_score = None

    if keep.sum().item() > 0:
        scores_keep = scores[keep]
        boxes_keep = boxes[keep]

        best_idx = scores_keep.argmax()
        best_score = scores_keep[best_idx].item()
        box = boxes_keep[best_idx].detach().cpu().numpy()

        # Convert cx,cy,w,h normalized box to x1,y1,x2,y2 pixels
        bx, by, bw, bh = box

        x1 = int((bx - bw / 2) * w)
        y1 = int((by - bh / 2) * h)
        x2 = int((bx + bw / 2) * w)
        y2 = int((by + bh / 2) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        best_box = [x1, y1, x2, y2]

    # Center lock circle
    cv2.circle(frame, (cx, cy), LOCK_RADIUS, (255, 255, 255), 2)
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 2)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        box_w = max(x2 - x1, 1)
        tx = (x1 + x2) // 2
        ty = (y1 + y2) // 2

        distance_m = (REAL_DRONE_WIDTH_M * FOCAL_LENGTH_PX) / box_w

        error_x = tx - cx
        error_y = ty - cy
        error = np.sqrt(error_x ** 2 + error_y ** 2)

        locked = error <= LOCK_RADIUS

        color = (0, 255, 0) if locked else (0, 0, 255)
        status = "TARGET LOCKED" if locked else "TARGET TRACKING"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.circle(frame, (tx, ty), 6, color, -1)
        cv2.line(frame, (cx, cy), (tx, ty), color, 2)

        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        cv2.putText(frame, f"Confidence: {best_score:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Distance: {distance_m:.2f} m", (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Offset X:{error_x}px Y:{error_y}px", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    else:
        cv2.putText(frame, "NO DRONE DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    out.write(frame)

cap.release()
out.release()

print("Done:", OUTPUT_PATH)