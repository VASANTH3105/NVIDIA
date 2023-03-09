import torch
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Load YOLOv5 model
weights = 'path/to/weights.pt'
device = select_device('')
model = attempt_load(weights, map_location=device)

# Define class labels
class_names = ['object_1', 'object_2', 'object_3']

# Define function to get center point of bounding box
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Load image and detect objects
img = cv2.imread('path/to/image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model(img, size=640)
results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)

# Loop over detected objects and get their center points
for result in results:
    if result is not None:
        for box in result[:, :4]:
            center = get_center(box)
            print(f"Object center: {center}")
