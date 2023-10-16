import json
import random
from PIL import Image

# Load the potential category mapping data
with open('potential_category_mapping_14.json', 'r') as f:
    potential_category_mapping = json.load(f)

# Load the ffair_annotation_original data
with open('ffair_annotation_original.json', 'r') as f:
    report_data = json.load(f)

# Extract cases from potential category mapping
mapped_cases = set()
for _, cases in potential_category_mapping.items():
    mapped_cases.update(cases)

# Extract cases from the "train" section of ffair_annotation_original
train_cases = set(report_data["train"].keys())

# Find cases in ffair_annotation_original that are not in potential category mapping
unmapped_cases = train_cases - mapped_cases

# Count of such cases
num_unmapped_cases = len(unmapped_cases)
print(num_unmapped_cases)

# Total number of cases in the "train" section of ffair_annotation_original
total_train_cases = len(train_cases)
print(total_train_cases)
ratio = num_unmapped_cases/total_train_cases
print(ratio)

# Reload the coco data, as we need to append to the original data without the previous additions
with open('coco_formatted_data.json', 'r') as f:
    coco_data = json.load(f)

# Get the highest id currently in coco_data to ensure uniqueness
highest_id = max([img["id"] for img in coco_data["images"]], default=0)
print("estimated number of images", highest_id/ratio)
numberOfCases = round((highest_id/ratio - 4540)/10)
print("number of cases ", numberOfCases)

# Convert the set to a list before sampling
selected_cases = random.sample(list(unmapped_cases), numberOfCases)

# Proceed with the previous logic to select images
selected_images = []

for case in selected_cases:
    case_images = report_data["train"][case]["Image_path"]
    if len(case_images) > 10:
        case_images = random.sample(case_images, 10)
    selected_images.extend(case_images)


# Construct image entries for the selected images
new_image_entries = []
for index, image_path in enumerate(selected_images):
    # Adjust the path to point to the FFAIR directory
    adjusted_image_path = "FFAIR/" + image_path
    
    # Open the image and get its dimensions
    with Image.open(adjusted_image_path) as img:
        width, height = img.size

    new_image_entries.append({
        "id": highest_id + index + 1, 
        "file_name": image_path,
        "width": width,
        "height": height
    })

# Add new images to the coco_data
coco_data["images"].extend(new_image_entries)

new_highest_id = max([img["id"] for img in coco_data["images"]])
print(f"New max ID: {new_highest_id}")


# Save the updated coco_formatted_data.json file
with open('updated_coco_formatted_data_v2.json', 'w') as f:
    json.dump(coco_data, f, indent=4)

# Return the number of images added and path to the updated file for verification
len(new_image_entries), 'updated_coco_formatted_data_v2.json'
