import json
import shutil
import os

# Load the COCO-formatted data
with open('final_coco_formatted_data.json', 'r') as f:
    coco_data = json.load(f)

# Ensure the destination directory exists
if not os.path.exists('detection_subset_new'):
    os.makedirs('detection_subset_new')

# Copy each image to the new directory and update the file names in the COCO data
for image_entry in coco_data["images"]:
    old_file_path = os.path.join('FFAIR', image_entry["file_name"])
    new_file_name = image_entry["file_name"].replace('/', '_')
    new_file_path = os.path.join('detection_subset_new', new_file_name)
    shutil.copy2(old_file_path, new_file_path)  # copy2 will copy both file data and metadata

    # Update filename in COCO data
    image_entry["file_name"] = new_file_name

# Save the updated COCO-formatted data
with open('updated_final_coco_formatted_data.json', 'w') as f:
    json.dump(coco_data, f, indent=4)

print("Images copied to 'detection_subset_new' and updated COCO-formatted data saved to 'updated_final_coco_formatted_data.json'")
