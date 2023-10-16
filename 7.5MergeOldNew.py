import json

# Load the content of the files
with open("lesion_info_original.json", "r") as f1:
    lesion_info_original = json.load(f1)

with open("filtered_results.json", "r") as f2:
    filtered_results = json.load(f2)

# Extract image names from the lesion_info_original dictionary
lesion_info_original_images = set(lesion_info_original.keys())

# Initialize a counter for new images added
new_images_added = 0

# Create a copy of the original dictionary to modify
merged_data = lesion_info_original.copy()

for img_name, details in filtered_results.items():
    short_name = img_name.split('/')[-2] + '/' + img_name.split('/')[-1]
    
    # If image name is not a duplicate, add it to merged_data in the desired format
    if short_name not in merged_data:
        # Convert data format from filtered_results to match that of lesion_info_original
        formatted_data = [[detail['category']] + detail['bbox'] for detail in details]
        merged_data[short_name] = formatted_data
        new_images_added += 1

# Save the merged data to a new JSON file
with open("path_to_merged_results_90.json", "w") as f3:
    json.dump(merged_data, f3)

print(f"Total new images added: {new_images_added}")
