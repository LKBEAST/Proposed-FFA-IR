import json
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the COCO data
with open("updated_final_coco_formatted_data.json", "r") as f:
    coco_data = json.load(f)

# Create a set of image IDs that have annotations
image_ids_with_annotations = set([anno['image_id'] for anno in coco_data['annotations']])

# Filter the images list to images that have annotations and images that don't
images_with_annotations = [img for img in coco_data['images'] if img['id'] in image_ids_with_annotations]
images_without_annotations = [img for img in coco_data['images'] if img['id'] not in image_ids_with_annotations]

# Create stratify labels for the images with annotations
image_to_categories = defaultdict(list)
for annotation in coco_data['annotations']:
    image_to_categories[annotation['image_id']].append(annotation['category_id'])

stratify_labels = [image_to_categories[img['id']][0] for img in images_with_annotations]

# Identify categories with only 1 instance
category_counts_for_split = defaultdict(int)
for label in stratify_labels:
    category_counts_for_split[label] += 1

single_instance_categories = {category for category, count in category_counts_for_split.items() if count == 1}

# Separate out these single instance images
single_instance_images = [img for img in images_with_annotations if image_to_categories[img['id']][0] in single_instance_categories]

for img in single_instance_images:
    images_with_annotations.remove(img)

# Refresh the stratify labels after removing single instance images
stratify_labels = [image_to_categories[img['id']][0] for img in images_with_annotations]

# Now perform the split
images_train_with_annotations, images_test_with_annotations = train_test_split(images_with_annotations, test_size=0.2, stratify=stratify_labels)
images_train_without_annotations, images_test_without_annotations = train_test_split(images_without_annotations, test_size=0.2)

# Merge the splits
images_train = images_train_with_annotations + images_train_without_annotations + single_instance_images
images_test = images_test_with_annotations + images_test_without_annotations

# Create coco_train and coco_test jsons
coco_train = {
    "images": images_train,
    "annotations": [anno for anno in coco_data['annotations'] if anno['image_id'] in [img['id'] for img in images_train]],
    "categories": coco_data['categories']
}

coco_test = {
    "images": images_test,
    "annotations": [anno for anno in coco_data['annotations'] if anno['image_id'] in [img['id'] for img in images_test]],
    "categories": coco_data['categories']
}

# Save the outputs
with open("coco_train_mod.json", "w") as f:
    json.dump(coco_train, f, indent = 4)

with open("coco_test_mod.json", "w") as f:
    json.dump(coco_test, f, indent = 4)

# Print the category that occurs least in coco_test
category_counts = defaultdict(int)
for annotation in coco_test['annotations']:
    category_counts[annotation['category_id']] += 1

min_category = min(category_counts, key=category_counts.get)
print(f"Category ID {min_category} has the minimum number of items: {category_counts[min_category]}")
