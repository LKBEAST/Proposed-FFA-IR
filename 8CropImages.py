import os
import json
from PIL import Image
from tqdm import tqdm

# Load the JSON data
with open('path_to_merged_results.json', 'r') as f:
    data = json.load(f)

# Dictionary of size combinations
combinations = [
    (768, 768), (3180, 2696), (512, 512), (384, 384), (3200, 2600), (3180, 2600), 
    (1600, 1200), (2448, 1956), (2285, 1900), (768, 576), (2124, 2056), 
    (1444, 1444), (1536, 1536), (1024, 1024), (2285, 1880), (2260, 1880),
    (3216, 2136), (1111, 1438), (1130, 1038), (1386, 1105), (1074, 1113)
]

def get_nearest_size(width, height):
    suitable_combinations = [(w, h) for w, h in combinations if w >= width and h >= height]
    if suitable_combinations:
        return min(suitable_combinations, key=lambda x: x[0]*x[1])
    else:
        return max(combinations, key=lambda x: x[0]*x[1])

if not os.path.exists('cropped_detected_objects_96'):
    os.makedirs('cropped_detected_objects_96')
    
# Iterate over FFAIR/case_x/y.jpeg files
base_dir = "FFAIR"
case_dirs = sorted([d for d in os.listdir(base_dir) if "case_" in d])[:30]  # take the first 10 cases
for case_dir in case_dirs:
    if "case_" in case_dir:
        for image_file in os.listdir(os.path.join(base_dir, case_dir)):
            img_path = os.path.join(base_dir, case_dir, image_file)
            
            key = f"{case_dir}/{image_file}"
            
            # Create target directory for this case if not exists
            new_dir = os.path.join('cropped_detected_objects_96', case_dir)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                
            try:
                # Check if image is in the JSON file
                if key in data:
                    detections = data[key]
                    for idx, detection in enumerate(detections):
                        bbox = [float(coord) for coord in detection[1:]]
                        with Image.open(img_path).convert('L') as img:
                            cropped_img = img.crop(bbox)
                            new_width, new_height = get_nearest_size(cropped_img.width, cropped_img.height)
                            # Create an all-black image with the new size
                            new_img = Image.new('L', (new_width, new_height), color=0)
                            # Paste the cropped image onto the new black image at the center
                            new_img.paste(cropped_img, ((new_width - cropped_img.width) // 2, (new_height - cropped_img.height) // 2))
                            # Modify the save path to append an index for multiple entries per image
                            new_img.save(os.path.join(new_dir, f"{os.path.splitext(image_file)[0]}_{idx}.jpeg"))
                else:
                    # If not in the JSON file, copy the whole image
                    with Image.open(img_path).convert('L') as img:
                        img.save(os.path.join(new_dir, image_file))
            except Exception as e:
                print(f"Error processing {img_path}. Error: {e}")