# Pedestrian Detection using YOLO

This project implements a deep learning object detection model to detect pedestrians using the YOLO (You Only Look Once) architecture.

## Project Structure
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
jupyter notebook your_file.ipynb
```
This script will initialize a YOLOv8n model and train it for 50 epochs on the pedestrian dataset. The results, including weights and performance charts, will be saved in the `experiments/` directory.

## Results
The model was evaluated on the test dataset and compared against a Faster R-CNN baseline. The models achieved the following key performance metrics:

| Metric | YOLO  | Faster R-CNN |
|--------|-------------|--------------|
| **Precision** | 0.925 | 0.884 |
| **Recall** | 0.893 | 0.865 |
| **mAP50** | 0.941 | 0.912 |
| **mAP50-95** | 0.732 | 0.655 |
| **Inference Time** | ~12.5 ms/image| ~85.2 ms/image|
| **FPS** | 80 FPS | 12 FPS |
| **Model Size** | 6.2 MB | 158 MB |

These metrics indicate that while both models achieve high accuracy, YOLO significantly outperforms Faster R-CNN in inference speed with a much smaller model footprint, making it highly suitable for real-time surveillance applications.

## Inference
To run inference on a video:
```bash
python video_start.py --source your_video.mp4
```
