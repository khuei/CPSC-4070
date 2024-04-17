from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = YOLO('detection-model.pt')

image_dir = 'images'
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    raise Exception("No image files found in the directory.")

random_image = random.choice(image_files)

results = model(random_image, classes=[0, 1])

result_len = len(results[0].boxes.cls.cpu().numpy())

fig, ax = plt.subplots(figsize=(10, 3))
rectangle_width = 10
x_limit = rectangle_width * result_len 
ax.set_xlim(0, x_limit if x_limit > 0 else rectangle_width)
ax.set_ylim(0, 20)
ax.set_title('Parking Lot Representation')
ax.axis('off')

x_offset = 0
for result in results:
    detect_boxes = result.boxes.data.cpu().numpy()
    sorted_boxes = sorted(detect_boxes, key=lambda b: b[0])

    for box in sorted_boxes:
        _, _, _, _, _, class_id = box
        label = 'Occupied' if class_id == 1 else 'Free'
        color = 'red' if class_id == 1 else 'green'
        rect = patches.Rectangle((x_offset, 5), rectangle_width, 10, linewidth=2, edgecolor=color, facecolor=color)
        ax.add_patch(rect)
        ax.text(x_offset + rectangle_width/2, 10, label, color='white', fontsize=12, ha='center', va='center')
        x_offset += rectangle_width

    result.show()

plt.show()
