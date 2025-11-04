import os
import json
import shutil

# Paths to organize
datasets = ["data/train", "data/valid", "data/test"]

for dataset_path in datasets:
    annotations_path = os.path.join(dataset_path, "_annotations.coco.json")

    # Skip if annotation file doesn't exist
    if not os.path.exists(annotations_path):
        print(f"⚠️ Skipping {dataset_path} (no annotations file found)")
        continue

    print(f"📁 Organizing {dataset_path}...")

    # Load annotation file
    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Create class folders
    for category in data["categories"]:
        class_name = category["name"]
        class_path = os.path.join(dataset_path, class_name)
        os.makedirs(class_path, exist_ok=True)

    # Move images into their class folders
    for item in data["annotations"]:
        image_id = item["image_id"]
        category_id = item["category_id"]

        # Find class name
        class_name = next(
            c["name"] for c in data["categories"] if c["id"] == category_id
        )

        # Find image name
        image_info = next(img for img in data["images"] if img["id"] == image_id)
        image_name = image_info["file_name"]

        src = os.path.join(dataset_path, image_name)
        dest = os.path.join(dataset_path, class_name, image_name)

        if os.path.exists(src):
            shutil.move(src, dest)

    print(f"✅ {dataset_path} organized successfully!\n")

print("🎉 All datasets organized into class folders!")
