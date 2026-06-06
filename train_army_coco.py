import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DeformableDetrForObjectDetection,
    DetrImageProcessor,
    TrainingArguments,
    Trainer,
)

# ================= PATHS =================
DATA_ROOT = "/teamspace/studios/this_studio"

DATASET_DIR = f"{DATA_ROOT}/armydataset"
TRAIN_DIR = f"{DATASET_DIR}/train"
VALID_DIR = f"{DATASET_DIR}/valid"

TRAIN_JSON = f"{TRAIN_DIR}/_annotations.coco.json"
VALID_JSON = f"{VALID_DIR}/_annotations.coco.json"

LAST_CHECKPOINT = f"{DATA_ROOT}/hf_detr_large_output_v2"
OUT_DIR = f"{DATA_ROOT}/hf_detr_army_output"

# ================= CHECK DATA =================
with open(TRAIN_JSON, "r") as f:
    train_coco = json.load(f)

with open(VALID_JSON, "r") as f:
    valid_coco = json.load(f)

categories = train_coco["categories"]

id2label = {cat["id"]: cat["name"] for cat in categories}
label2id = {cat["name"]: cat["id"] for cat in categories}
num_labels = len(categories)

print("========== DATASET INFO ==========")
print("Train images:", len(train_coco["images"]))
print("Train annotations:", len(train_coco["annotations"]))
print("Valid images:", len(valid_coco["images"]))
print("Valid annotations:", len(valid_coco["annotations"]))
print("Categories:", categories)
print("Number of classes:", num_labels)
print("id2label:", id2label)
print("==================================")

# ================= DATASET CLASS =================
class CocoArmyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        with open(annotation_file, "r") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]

        self.image_id_to_annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            self.image_id_to_annotations.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = os.path.join(self.image_dir, file_name)

        image = Image.open(image_path).convert("RGB")

        annotations = self.image_id_to_annotations.get(image_id, [])

        target = {
            "image_id": image_id,
            "annotations": annotations,
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# ================= LOAD PROCESSOR + MODEL =================
print("Loading checkpoint:", LAST_CHECKPOINT)

processor = DetrImageProcessor.from_pretrained(LAST_CHECKPOINT)

model = DeformableDetrForObjectDetection.from_pretrained(
    LAST_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

model.config.id2label = id2label
model.config.label2id = label2id
model.config.num_labels = num_labels

# ================= DATASETS =================
train_dataset = CocoArmyDataset(TRAIN_DIR, TRAIN_JSON, processor)
valid_dataset = CocoArmyDataset(VALID_DIR, VALID_JSON, processor)

# ================= COLLATE FUNCTION =================
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")

    labels = [item["labels"] for item in batch]

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }

# ================= TRAINING ARGS =================
training_args = TrainingArguments(
    output_dir=OUT_DIR,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,

    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=1e-4,
    warmup_ratio=0.05,

    logging_steps=25,

    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,

    eval_strategy="steps",
    eval_steps=1000,

    remove_unused_columns=False,
    fp16=True,

    dataloader_num_workers=4,
    report_to="none",

    load_best_model_at_end=False,
)

# ================= TRAINER =================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    tokenizer=processor,
)

# ================= TRAIN =================
print("Starting training...")
trainer.train()

# ================= SAVE =================
trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)

print("Training completed.")
print("Saved model to:", OUT_DIR)
