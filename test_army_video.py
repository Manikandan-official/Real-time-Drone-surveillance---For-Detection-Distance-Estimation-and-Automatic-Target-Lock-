import cv2
import torch
from PIL import Image
from transformers import DeformableDetrForObjectDetection, DetrImageProcessor

# ================= PATHS =================
VIDEO_PATH = "test1.mp4"
MODEL_PATH = "/teamspace/studios/this_studio/hf_detr_army_output"
OUT_PATH = "army_output.mp4"

CONF_THRESH = 0.20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ================= LOAD MODEL =================
processor = DetrImageProcessor.from_pretrained(MODEL_PATH)

model = DeformableDetrForObjectDetection.from_pretrained(
    MODEL_PATH
).to(device)

model.eval()

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0:
    fps = 25

out = cv2.VideoWriter(
    OUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

frame_count = 0

print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)

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

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):

        score = score.item()
        label = label.item()

        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)

        class_name = model.config.id2label.get(label, str(label))

        # Different colors
        if class_name == "Shahed136":
            color = (0, 0, 255)

        elif class_name == "orlan":
            color = (0, 255, 0)

        else:
            color = (255, 0, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        text = f"{class_name}: {score:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    out.write(frame)

cap.release()
out.release()

print("Done.")
print("Saved output:", OUT_PATH)