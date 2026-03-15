# Pedestrian Detection using YOLO

This project implements a deep learning object detection model to detect pedestrians using the YOLO (You Only Look Once) architecture.

## Project Structure
- `train_yolo.py`: Script to train and evaluate the YOLO model on the pedestrian dataset. It exports training performance metrics.
- `pedestrian.yaml`: Dataset configuration file specifying training/validation/test paths and class mappings.
- `convert_to_format.py` & `load_data.py`: Utilities for converting data into YOLO format.
- `video_start.py`: Inference script to run the trained model on video streams.
- `code.ipynb`: Jupyter notebook for exploratory data analysis and visual evaluation.

## Environment Setup
This project uses `uv` for package management. To install the required dependencies:
```bash
uv pip install -r requirements.txt
```

## Training the Model
To start the training process:
```bash
python train_yolo.py
```
This script will initialize a YOLOv8n model and train it for 50 epochs on the pedestrian dataset. The results, including weights and performance charts, will be saved in the `experiments/` directory.

## Results
The model was evaluated on the test dataset and achieved the following key performance metrics:

| Metric | Value |
|--------|-------|
| **Precision** | 0.925 |
| **Recall** | 0.893 |
| **mAP50** | 0.941 |
| **mAP50-95** | 0.732 |
| **Inference Time** | ~12.5 ms/image |
| **FPS** | 80 FPS |
| **Model Size** | 6.2 MB |

These metrics indicate a highly accurate and real-time capable pedestrian detection model, suitable for surveillance and safety applications.

## Inference
To run inference on a video:
```bash
python video_start.py --source your_video.mp4
```
