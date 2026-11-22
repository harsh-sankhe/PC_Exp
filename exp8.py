# First install in VS Code terminal: pip install ultralytics
import cv2
from ultralytics import YOLO

image_path = "image.jpg"
img = cv2.imread(image_path)
if img is None: raise FileNotFoundError("Put image.jpg in this folder")

model = YOLO("yolo11n.pt")  # downloads model on first run
result = model(image_path)[0]
annotated = result.plot()

print("Detected Objects and Bounding Boxes:")
for box in result.boxes:
    class_id = int(box.cls[0])
    name = model.names[class_id]
    confidence = float(box.conf[0])
    x1,y1,x2,y2 = box.xyxy[0].tolist()
    print(f"{name}: {confidence:.2f} | Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

cv2.imshow("Original",img)
cv2.imshow("YOLO Object Detection",annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
