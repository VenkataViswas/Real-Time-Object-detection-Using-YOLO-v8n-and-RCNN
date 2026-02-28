import time
import os
import pandas as pd
from ultralytics import YOLO

DATA_YAML = "pedestrian.yaml"

def train_yolo():

    start_time = time.time()

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        project="experiments",
        name="yolo_run"
    )

    train_time = time.time() - start_time

    print("Training time:", train_time)

    return model, train_time


def evaluate_yolo(model):

    metrics = model.val(data=DATA_YAML)

    precision = metrics.box.p
    recall = metrics.box.r
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    return precision, recall, map50, map5095


def measure_speed(model):

    test_dir = "dataset/images/test"
    images = os.listdir(test_dir)[:50]

    start = time.time()

    for img in images:
        model.predict(os.path.join(test_dir, img), verbose=False)

    inference_time = (time.time() - start) / len(images)

    fps = 1 / inference_time

    return inference_time, fps


if __name__ == "__main__":

    model, train_time = train_yolo()

    precision, recall, map50, map5095 = evaluate_yolo(model)

    inf_time, fps = measure_speed(model)

    model_size = os.path.getsize("experiments/yolo_run/weights/best.pt") / (1024*1024)

    results = {
        "Model": "YOLO",
        "Precision": precision,
        "Recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
        "Training_Time_sec": train_time,
        "Inference_Time_sec": inf_time,
        "FPS": fps,
        "Model_Size_MB": model_size
    }

    df = pd.DataFrame([results])
    df.to_csv("yolo_results.csv", index=False)

    print(df)
