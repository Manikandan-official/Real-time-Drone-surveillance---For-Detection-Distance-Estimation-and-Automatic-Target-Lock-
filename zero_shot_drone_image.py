import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ================= PATHS =================
IMAGE_PATH = "09.jpg"          # change image name
OUT_PATH = "zero_shot_drone.jpg"

# ================= PROMPT =================
TEXT_PROMPT = "drone. military drone. UAV. flying aircraft."

# ================= MODEL =================
MODEL_ID = "IDEA-Research/grounding-dino-base"

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()

# ================= IMAGE =================
image = Image.open(IMAGE_PATH).convert("RGB")
frame = cv2.imread(IMAGE_PATH)

h, w = frame.shape[:2]

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

print("Detected:", len(boxes))

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

cv2.imwrite(OUT_PATH, frame)
print("Saved:", OUT_PATH)
