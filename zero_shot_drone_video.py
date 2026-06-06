import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ================= PATHS =================
VIDEO_PATH = "4.mp4"
OUT_PATH = "zero_shot_drone_output.mp4"

# ================= PROMPT =================
TEXT_PROMPT = "drone. military drone. UAV. flying aircraft."

MODEL_ID = "IDEA-Research/grounding-dino-base"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()

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

    frame_id += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(h, w)]
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    if len(boxes) == 0:
        cv2.putText(
            frame,
            "NO DRONE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
        score = score.item()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.putText(
            frame,
            f"{label}: {score:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    out.write(frame)

    if frame_id % 20 == 0:
        print("Processed:", frame_id)

cap.release()
out.release()

print("Saved:", OUT_PATH)