import os
import shutil

# Define the path to the original and new labels folders
original_labels_folder = "/data/tml/lymonet/lymo_yolo_unclipped/labels"
new_labels_folder = "/data/tml/lymonet/lymo_yolo_0/labels"

# Create the new folder if it doesn't exist
if not os.path.exists(new_labels_folder):
    os.makedirs(new_labels_folder)

# Define the classes you want to change
old_classes = ["0", "1", "2"]
new_class = "0"

# Iterate through the train, test, and val folders
split_folders = ["train", "test", "val"]

for folder in split_folders:
    original_folder_path = os.path.join(original_labels_folder, folder)
    new_folder_path = os.path.join(new_labels_folder, folder)
    
    # Create the new subfolder if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    # Iterate through each file in the folder and modify the class
    for file_name in os.listdir(original_folder_path):
        if file_name.endswith(".txt"):
            label_path = os.path.join(original_folder_path, file_name)
            new_label_path = os.path.join(new_folder_path, file_name)
            
            with open(label_path, 'r') as label_file:
                labels = label_file.readlines()
            
            modified_labels = []
            for label in labels:
                class_id, x, y, width, height = label.split()
                # Modify the class to the new class
                class_id = str(old_classes.index(class_id) if class_id in old_classes else class_id)
                modified_labels.append(f"{new_class} {x} {y} {width} {height}")
            
            # Write the modified labels to the new file
            with open(new_label_path, 'w') as label_file:
                label_file.write("\n".join(modified_labels))

