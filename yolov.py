from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8s model (trained on COCO)
model = YOLO('yolov8m.pt')
print("Model loaded successfully.")

# Detect objects in an image
img = cv2.imread('C:/Users/Lenovo Loq/Pictures/ground floor/frame_600.jpg') # Replace with your image path
results = model(img)

# # Get boxes
# person_count = 0
# for box in results[0].boxes:
#     cls_id = int(box.cls[0])
#     conf = box.conf[0].item()
    
#     if cls_id == 56:  # class 0 = person
#         person_count += 1
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         label = f"Chair {conf:.2f}"
        
#         # Draw box and label
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# # Display with total count in the window title
# window_name = f"Person Detection - Total Persons: {person_count}"
# cv2.namedWindow("Output", cv2.WINDOW_NORMAL)  # allow resizing
# cv2.imshow("Output", img)
# cv2.resizeWindow("Output", 1280, 720)         # manually set window size
# # results[0].show() # display all detections with labels
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, stream=True)
#     for r in results:
#         im = r.plot()  # draw boxes
#         cv2.imshow("YOLOv11s COCO", im)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# ***Marking only unoccupied chairs***
chairs = []
people = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:  # person
            people.append((x1, y1, x2, y2))
        elif cls == 56:  # chair
            chairs.append((x1, y1, x2, y2))

def iou(boxA, boxB):
    # Calculate Intersection over Union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


occupied_chairs = []

for chair in chairs:
    occupied = False
    for person in people:
        if iou(chair, person) > 0.1:  #  tune this threshold to adjust confidence
            occupied = True
            occupied_chairs.append(chair)
            break
        


for (x1, y1, x2, y2) in occupied_chairs:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "Free Chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.putText(img, f"Free chairs: {len(occupied_chairs)}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

cv2.namedWindow("Free Chairs", cv2.WINDOW_NORMAL)  # allow resizing
cv2.resizeWindow("Free Chairs", 1280, 720)  # manually set window size
cv2.imshow("Free Chairs", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

