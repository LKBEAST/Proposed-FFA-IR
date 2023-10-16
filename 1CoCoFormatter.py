import json
import os
from PIL import Image

def convert_to_coco_format(lesion_info_file, image_directory, output_file):
    # Load the lesion_info_original.json data
    with open(lesion_info_file, 'r') as json_file:
        lesion_info = json.load(json_file)

    # Initialise COCO formatted data
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create a list of unique categories from the lesion labels
    unique_categories = set()
    for boxes in lesion_info.values():
        for box in boxes:
            unique_categories.add(box[0])
    category_id_map = {name: idx for idx, name in enumerate(sorted(unique_categories), start=1)}
    coco_data["categories"] = [{"id": idx, "name": name} for name, idx in category_id_map.items()]

    # Extract image properties and annotations
    annotation_id = 1
    for image_id, (key, boxes) in enumerate(lesion_info.items()):
        # Extract image properties
        image_path = os.path.join(image_directory, key)
        with Image.open(image_path) as img:
            width, height = img.size

        coco_data["images"].append({
            "id": image_id,
            "file_name": key,
            "width": width,
            "height": height
        })

        # Extract annotations
        for box in boxes:
            category, x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_map[category],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id += 1

    # Write the COCO formatted data to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"Data in COCO format has been saved to {output_file}")

convert_to_coco_format('lesion_info_original.json', 'FFAIR', 'coco_formatted_data.json')
