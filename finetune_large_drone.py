import os, json, torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DeformableDetrForObjectDetection,
    DetrImageProcessor,
    TrainingArguments,
    Trainer
)

# ---------------- PATHS ----------------
DATA_ROOT = "/teamspace/studios/this_studio"
COCO_JSON_DIR = f"{DATA_ROOT}/drone_coco"
IMAGE_ROOT = f"{DATA_ROOT}/drone_split"

BASE_CHECKPOINT = f"{DATA_ROOT}/hf_detr_large_output"
OUT_DIR = f"{DATA_ROOT}/hf_detr_large_output_v2"

# ---------------- DATASET ----------------
class CocoDroneDataset(Dataset):
    def __init__(self, split, processor):
        self.processor = processor
        ann_path = f"{COCO_JSON_DIR}/instances_{split}.json"
        img_dir = f"{IMAGE_ROOT}/{split}"

        with open(ann_path, "r") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}

        anns_by_img = {}
        for ann in coco["annotations"]:
            ann["category_id"] = 0
            ann["iscrowd"] = ann.get("iscrowd", 0)
            ann["area"] = ann.get("area", ann["bbox"][2] * ann["bbox"][3])
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self.records = []
        for img_id, img in images.items():
            anns = anns_by_img.get(img_id, [])
            if len(anns) == 0:
                continue

            self.records.append({
                "image_id": img_id,
                "file_name": img["file_name"],
                "img_dir": img_dir,
                "annotations": anns
            })

        # FAST MODE: use subset first
        if split == "train":
            self.records = self.records[:10000]
        elif split == "val":
            self.records = self.records[:1000]

        print(f"{split}: {len(self.records)} images loaded")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(rec["img_dir"], "images", rec["file_name"])

        image = Image.open(img_path).convert("RGB")

        target = {
            "image_id": rec["image_id"],
            "annotations": rec["annotations"]
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0]
        }

# ---------------- COLLATE ----------------
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch]
    }

# ---------------- MODEL ----------------
processor = DetrImageProcessor.from_pretrained(
    BASE_CHECKPOINT,
    size={"shortest_edge": 480, "longest_edge": 640}
)

model = DeformableDetrForObjectDetection.from_pretrained(
    BASE_CHECKPOINT,
    num_labels=1,
    ignore_mismatched_sizes=True
)

# ---------------- DATA ----------------
train_dataset = CocoDroneDataset("train", processor)
val_dataset = CocoDroneDataset("val", processor)

# ---------------- TRAINING ----------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-5,
    weight_decay=1e-4,
    gradient_accumulation_steps=4,
    logging_steps=100,
    save_steps=10000,
    eval_steps=10000,
    eval_strategy="no",
    save_total_limit=2,
    fp16=True,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn
)

trainer.train()

trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)

print("Fine-tuning finished.")
print("Saved to:", OUT_DIR)