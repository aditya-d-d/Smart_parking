from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('best.pt')

# Load video
cap = cv2.VideoCapture("1057865701-preview.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)[0]

    # Draw results
    annotated_frame = results.plot()
    cv2.imshow("Parking Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
