import cv2
from ultralytics import YOLO

# load trained YOLO model
model = YOLO("C:/Users/Srinivas/Desktop/deep learning/runs/detect/train/weights/best.pt")

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run detection
    results = model(frame)

    # draw detections
    annotated_frame = results[0].plot()

    # show frame
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()