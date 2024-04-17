#!/usr/bin/env python3

from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def plot_diagram(results, result_len, output_path):
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
            ax.text(x_offset + rectangle_width / 2, 10, label, color='white', fontsize=12, ha='center', va='center')
            x_offset += rectangle_width
    plt.savefig(output_path)
    plt.close(fig)

def combine_images(img1_path, img2_path, output_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    width = img1.width + img2.width
    height = max(img1.height, img2.height)

    new_img = Image.new('RGB', (width, height))
    img1_offset = (height - img1.height) // 2  # Calculate vertical offset to center the diagram
    new_img.paste(img1, (0, img1_offset))
    new_img.paste(img2, (img1.width, 0))

    new_img.save(output_path)

model = YOLO('detection-model.pt')
image_dir = 'images'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    raise Exception("No image files found in the directory.")

for image_path in image_files:
    base_filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_filename)[0]
    individual_output_dir = os.path.join(output_dir, name_without_ext)
    os.makedirs(individual_output_dir, exist_ok=True)

    results = model(image_path, classes=[0, 1])
    result_len = len(results[0].boxes.cls.cpu().numpy())

    diagram_path = os.path.join(individual_output_dir, f"{name_without_ext}_diagram.png")
    yolo_output_path = os.path.join(individual_output_dir, f"{name_without_ext}_yolo_output.png")
    combined_output_path = os.path.join(individual_output_dir, f"{name_without_ext}_combined.png")

    plot_diagram(results, result_len, diagram_path)
    results[0].save(yolo_output_path)  # Save the YOLO output
    combine_images(diagram_path, yolo_output_path, combined_output_path)
