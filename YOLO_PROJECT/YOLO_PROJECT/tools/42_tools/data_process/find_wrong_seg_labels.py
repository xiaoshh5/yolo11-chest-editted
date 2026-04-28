import os

def find_bounding_boxes(annotation_folder):
    # List to store image filenames with bounding box annotations
    bounding_box_image_filenames = []

    # Loop over all annotation files in the folder
    for annotation_file in os.listdir(annotation_folder):
        if annotation_file.endswith('.txt'):
            with open(os.path.join(annotation_folder, annotation_file), 'r') as f:
                lines = f.readlines()

            # Check each line in the annotation file
            for line in lines:
                # Split the line to get the elements
                elements = line.strip().split()

                # YOLOv8 object detection (bounding box) should have 5 elements: class_id, x_center, y_center, width, height
                # Polygon annotations will have more elements (for instance segmentation)
                if len(elements) == 5:
                    bounding_box_image_filenames.append(annotation_file.replace('.txt', '.jpg'))  # Assuming image is .jpg

    return bounding_box_image_filenames


# Define the path to your YOLOv8 annotations folder
annotation_folder = r'D:/CHW/Archaeological_images/prepare_test/Dataset/labels/test'

# Find all images with bounding box annotations
bounding_box_image_filenames = find_bounding_boxes(annotation_folder)

print("Found bounding boxes in the following images:", bounding_box_image_filenames)
import os.path  as osp
import shutil
# AUTO REMOVE
for bounding_box_image_filename in bounding_box_image_filenames:
    image_path = osp.join(annotation_folder.replace("labels", "images"), bounding_box_image_filename.replace('.jpg', ".png"))
    label_path = osp.join(annotation_folder, bounding_box_image_filename.replace('.jpg', ".txt"))
    print(
        image_path
    )
    print(osp.isfile(image_path))
    print(osp.isfile(label_path))
    try:
        os.remove(image_path)


        os.remove(label_path)
    except:
        print("拉了")