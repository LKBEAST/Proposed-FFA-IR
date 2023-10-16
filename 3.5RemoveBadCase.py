import json

# Reload the updated COCO formatted data
with open('updated_coco_formatted_data_v2.json', 'r') as f:
    coco_data = json.load(f)

# Get all image IDs that start with "case_7293"
to_remove_image_ids = [img["id"] for img in coco_data["images"] if "case_7293" in img["file_name"]]

# Remove these images from the dataset
coco_data["images"] = [img for img in coco_data["images"] if img["id"] not in to_remove_image_ids]

# Remove annotations associated with these images
coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann["image_id"] not in to_remove_image_ids]

# Reorder all the IDs for continuity
# Images
for idx, image in enumerate(coco_data["images"], start=1):
    image["id"] = idx

# Annotations
for idx, ann in enumerate(coco_data["annotations"], start=1):
    ann["id"] = idx

# Save the updated data
updated_file_path = 'final_coco_formatted_data.json'
with open(updated_file_path, 'w') as f:
    json.dump(coco_data, f, indent=4)

updated_file_path
