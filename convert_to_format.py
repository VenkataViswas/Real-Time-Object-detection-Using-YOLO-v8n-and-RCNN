import os
import shutil
import pandas as pd
import json

source_path = "data/Pedestrian Detection.v1i.tensorflow"
target_path = "dataset"

splits = {
    "train": "train",
    "valid": "val",
    "test": "test"
}

# Create folders
for split in splits.values():
    os.makedirs(f"{target_path}/images/{split}", exist_ok=True)
    os.makedirs(f"{target_path}/labels/{split}", exist_ok=True)

os.makedirs(f"{target_path}/rcnn_annotations", exist_ok=True)

img_id = 1
ann_id = 1

for src_split, dst_split in splits.items():

    split_path = os.path.join(source_path, src_split)
    csv_path = os.path.join(split_path, "_annotations.csv")

    df = pd.read_csv(csv_path)

    images = []
    annotations = []

    for filename in df["filename"].unique():

        rows = df[df["filename"] == filename]

        src_img = os.path.join(split_path, filename)
        dst_img = os.path.join(target_path, "images", dst_split, filename)

        if not os.path.exists(src_img):
            continue

        shutil.copy(src_img, dst_img)

        width = int(rows.iloc[0]["width"])
        height = int(rows.iloc[0]["height"])

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        label_file = os.path.join(
            target_path,
            "labels",
            dst_split,
            os.path.splitext(filename)[0] + ".txt"
        )

        with open(label_file, "w") as f:

            for _, row in rows.iterrows():

                xmin = int(row["xmin"])
                ymin = int(row["ymin"])
                xmax = int(row["xmax"])
                ymax = int(row["ymax"])

                # YOLO format
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                f.write(
                    f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
                )

                # COCO format for Faster R-CNN
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })

                ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "pedestrian"}
        ]
    }

    json_path = os.path.join(
        target_path,
        "rcnn_annotations",
        f"{dst_split}.json"
    )

    with open(json_path, "w") as f:
        json.dump(coco, f, indent=4)

print("Dataset prepared successfully for YOLO and Faster R-CNN")