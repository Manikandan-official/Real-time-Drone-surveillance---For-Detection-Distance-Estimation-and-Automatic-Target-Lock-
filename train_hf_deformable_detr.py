import os, json, torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    TrainingArguments,
    Trainer
)

DATA_ROOT = "/teamspace/studios/this_studio/coco_drone"
OUT_DIR = "/teamspace/studios/this_studio/hf_detr_output"


class CocoDroneDataset(Dataset):
    def __init__(self, split, processor):
        self.split = split
        self.processor = processor

        ann_path = f"{DATA_ROOT}/annotations/instances_{split}2017.json"
        img_dir = f"{DATA_ROOT}/{split}2017"

        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.img_dir = img_dir
        self.images = {img["id"]: img for img in coco["images"]}

        anns_by_img = {}
        for ann in coco["annotations"]:
            ann["category_id"] = 0
            ann["iscrowd"] = ann.get("iscrowd", 0)
            ann["area"] = ann.get("area", ann["bbox"][2] * ann["bbox"][3])
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self.records = []
        for img_id, img in self.images.items():
            anns = anns_by_img.get(img_id, [])
            if len(anns) == 0:
                continue
            self.records.append({
                "image_id": img_id,
                "file_name": img["file_name"],
                "annotations": anns
            })

        print(f"{split}: {len(self.records)} images loaded")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = os.path.join(self.img_dir, rec["file_name"])
        image = Image.open(image_path).convert("RGB")

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


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")

model = DeformableDetrForObjectDetection.from_pretrained(
    "SenseTime/deformable-detr",
    num_labels=1,
    id2label={0: "drone"},
    label2id={"drone": 0},
    ignore_mismatched_sizes=True
)

train_dataset = CocoDroneDataset("train", processor)
val_dataset = CocoDroneDataset("val", processor)

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=30,
    learning_rate=2e-5,
    weight_decay=1e-4,
    logging_steps=20,
    save_steps=300,
    eval_steps=300,
    eval_strategy="steps",
    save_total_limit=2,
    fp16=True,
    remove_unused_columns=False,
    dataloader_num_workers=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

trainer.train()
trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)

print("Training finished. Saved to:", OUT_DIR)
