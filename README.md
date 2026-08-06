# Workshop 2 - Object detection using web camera

## Name : Loknaath P
## Register number : 212223240080

## Aim
To perform real-time object detection using a pre-trained YOLOv4 (You Only Look Once Version 4) model through the laptop webcam.

## Algorithm
1. Import the required libraries: Ultralytics YOLO, OpenCV, Matplotlib, and IPython display.
2. Load the pre-trained YOLOv8m model using YOLO("yolov8m.pt").
3. Initialize the laptop webcam using cv2.VideoCapture(0).
4. Check whether the webcam is opened successfully.
5. Capture video frames continuously from the webcam.
6.Pass each captured frame to the YOLOv8 model with a confidence threshold of 0.60.
7. Detect objects present in the frame.
8. Draw bounding boxes, class names, and confidence scores using results[0].plot().
9. Convert the annotated frame from BGR to RGB format for Matplotlib display.
10. Continuously update the output in the Jupyter Notebook using clear_output().
11. Stop the detection when the specified exit key is pressed or the webcam is closed.
12. Release the webcam and close all OpenCV windows.

## Program
```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
```
```python
# Load a more accurate YOLO model
model = YOLO("yolov8m.pt")
```
```python
# Open webcam
cap = cv2.VideoCapture(0)
```
```python
# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(
            frame,
            conf=0.60,       
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Matplotlib
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display in Jupyter
        clear_output(wait=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_frame)
        plt.title("Real-Time Object Detection")
        plt.axis("off")
        plt.show()

# Release webcam
cap.release()
cv2.destroyAllWindows()
```
## Output
<img width="912" height="701" alt="image" src="https://github.com/user-attachments/assets/9abc2e3b-8dc1-4a48-8e74-cbd5b097716d" />


## Result
The pre-trained YOLOv8m model successfully detected and classified objects from the laptop webcam in real time. The detected objects were displayed with bounding boxes, class labels, and confidence scores, demonstrating accurate and efficient real-time object detection.
