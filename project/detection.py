from ultralytics import YOLO
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load a pretrained YOLOv8n model
model = YOLO('detection-model.pt')

# Get all image files from the 'images' directory
image_dir = 'images'
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Ensure there are images to choose from
if not image_files:
    raise Exception("No image files found in the directory.")

# Pick one random image from the list
random_image = random.choice(image_files)

# Run inference and filter for classes 0 and 1
results = model(random_image, classes=[0, 1])

result_len = len(results[0].boxes.cls.cpu().numpy())

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 3))
# Calculate the required x_limit based on the number of results and rectangle width
rectangle_width = 10  # Width of each rectangle
x_limit = rectangle_width * result_len  # No extra space between rectangles
ax.set_xlim(0, x_limit if x_limit > 0 else rectangle_width)  # Ensure there is at least space for one box
ax.set_ylim(0, 20)
ax.set_title('Parking Lot Representation')
ax.axis('off')  # Hide the axes

# Draw the boxes for each detected object
x_offset = 0  # Start offset for the first box
for result in results:
    # Extract boxes and sort them by the x1 coordinate
    detect_boxes = result.boxes.data.cpu().numpy()
    sorted_boxes = sorted(detect_boxes, key=lambda b: b[0])  # Sort boxes by x1 coordinate
    
    # Draw each box
    for box in sorted_boxes:
        _, _, _, _, _, class_id = box
        label = 'Occupied' if class_id == 1 else 'Free'
        color = 'red' if class_id == 1 else 'green'
        rect = patches.Rectangle((x_offset, 5), rectangle_width, 10, linewidth=2, edgecolor=color, facecolor=color)
        ax.add_patch(rect)
        ax.text(x_offset + rectangle_width/2, 10, label, color='white', fontsize=12, ha='center', va='center')
        x_offset += rectangle_width  # Move to the next position for the next box; rectangles touch each other

    result.show()

# Display the plot
plt.show()
