import json
import os

# Load the JSON file
with open("ffair_annotation_original.json", 'r') as file:
    data = json.load(file)

# Base directory where the cases' directories are located (change if necessary)
base_directory = "."

# Extract the train section
train_data = data.get('train', {})

# Loop through each case in train_data
for case, details in train_data.items():
    # Extract image paths for the current case
    image_paths = details.get('Image_path', [])
    
    # Get the directory for the current case
    case_directory = os.path.join(base_directory, case)
    
    # Check if the directory exists
    if os.path.exists(case_directory):
        # List the files in the directory for the current case
        directory_files = os.listdir(case_directory)
        
        # Determine the missing files in Image_path and extra files in the directory
        missing_in_image_path = [file for file in image_paths if file not in directory_files]
        extra_in_directory = [file for file in directory_files if file not in image_paths]
        
        # Update the image_paths list
        for file in missing_in_image_path:
            image_paths.remove(file)
        for file in extra_in_directory:
            image_paths.append(os.path.join(case, file))
        
        # Update the Image_path in the data
        details['Image_path'] = image_paths

# Save the updated data to a new JSON file
with open("ffair_annotation_updated.json", 'w') as file:
    json.dump(data, file, indent=4)

print("Annotations updated and saved to 'ffair_annotation_updated.json'")
