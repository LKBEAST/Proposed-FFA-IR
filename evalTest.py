import json
import detectron2
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os

# Load the COCO JSON file to get class names
with open("updated_final_coco_formatted_data.json", 'r') as f:
    coco_data = json.load(f)
class_names = [category['name'] for category in coco_data['categories']]

# Register your datasets in COCO format
register_coco_instances("my_dataset_train", {}, "coco_train.json", "detection_subset_new")
register_coco_instances("my_dataset_test", {}, "coco_test.json", "detection_subset_new")

MetadataCatalog.get("my_dataset_train").set(thing_classes=class_names)
MetadataCatalog.get("my_dataset_test").set(thing_classes=class_names)

# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "./outputGOOD/model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (10000, 12500)  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 46
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

# Create output directory
os.makedirs("./outputGOOD/", exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluation
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./outputGOOD")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(trainer.model, val_loader, evaluator)

import json
from detectron2.structures import BoxMode, Boxes
from detectron2.utils.visualizer import _create_text_labels
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the COCO JSON file
with open("coco_test.json", "r") as f:
    coco_data = json.load(f)

# Load the predicted data
with open("outputGOOD/coco_instances_results.json", "r") as f:
    predicted_data = json.load(f)

# List of top four categories
top_categories = ["6", "14", "10", "19"]

# Convert category names to category IDs
top_category_ids = [next(cat['id'] for cat in coco_data['categories'] if cat['name'] == category_name) for category_name in top_categories]

# Set the threshold to 0.9
threshold = 0.9

for category_id_of_interest, category_name in zip(top_category_ids, top_categories):
    # Filter out predictions for the current category based on threshold
    thresholded_predictions = [pred for pred in predicted_data if pred["category_id"] == category_id_of_interest and pred["score"] >= threshold]
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    misidentified_as_other_category = 0
    misidentified_as_no_category = 0

    for d in coco_data["images"]:
        image_id = d["id"]
        
        # Ground truth boxes for the category of interest
        gt_boxes = [ann["bbox"] for ann in coco_data["annotations"] if ann["image_id"] == image_id and ann["category_id"] == category_id_of_interest]
        pred_boxes = [pred["bbox"] for pred in thresholded_predictions if pred["image_id"] == image_id]

        # If there are no ground truth boxes or predicted boxes for this image, update FP and FN and skip to next iteration
        if not gt_boxes and not pred_boxes:
            continue  # Nothing to do here
        if not gt_boxes:
            false_positives += len(pred_boxes)
            continue
        if not pred_boxes:
            false_negatives += len(gt_boxes)
            continue

        # Convert coco bbox format [x,y,width,height] to [x1,y1,x2,y2]
        gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]
        pred_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in pred_boxes]

        gt_boxes = Boxes(gt_boxes)
        pred_boxes = Boxes(pred_boxes)

        iou_matrix = detectron2.structures.pairwise_iou(gt_boxes, pred_boxes)

        matched_gt = []
        matched_pred = []

        for i, j in enumerate(iou_matrix.argmax(dim=1)):
                if iou_matrix[i, j] >= 0.5:
                    true_positives += 1
                    matched_gt.append(i)
                    matched_pred.append(j)

        false_positives += len(pred_boxes) - len(matched_pred)
        false_negatives += len(gt_boxes) - len(matched_gt)
        # Create a list of unmatched predicted boxes
        # Create a list of unmatched predicted boxes
        unmatched_pred_boxes = [box for idx, box in enumerate(pred_boxes) if idx not in matched_pred]

        # Now, only consider these unmatched boxes for the misidentification calculations
        for pred in unmatched_pred_boxes:
            gt_boxes_for_image = [ann["bbox"] for ann in coco_data["annotations"] if ann["image_id"] == image_id]
            gt_boxes_for_image = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes_for_image]
            gt_boxes_for_image = Boxes(gt_boxes_for_image)

            pred_tensor = torch.tensor(pred) if not isinstance(pred, torch.Tensor) else pred
            if len(pred_tensor.shape) == 1:
                pred_boxes_2d = pred_tensor.unsqueeze(0)  # Convert 1D tensor to 2D tensor with shape (1, 4)
            else:
                pred_boxes_2d = pred_tensor

            iou_matrix = detectron2.structures.pairwise_iou(gt_boxes_for_image, Boxes(pred_boxes_2d))

            max_iou = iou_matrix.max()
            if max_iou >= 0.5:
                matching_gt = coco_data["annotations"][iou_matrix.argmax()]
                if matching_gt["category_id"] != category_id_of_interest:
                    misidentified_as_other_category += 1
            else:
                misidentified_as_no_category += 1

        # Calculate Precision and Recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    print(f"For category {category_name}:")
    print("Precision at threshold 0.9:", precision)
    print("Recall at threshold 0.9:", recall)
    print(f"Percentage misidentified as another category: {misidentified_as_other_category / false_positives * 100 if false_positives > 0 else 0:.2f}%")
    print(f"Percentage misidentified with no category: {misidentified_as_no_category / false_positives * 100 if false_positives > 0 else 0:.2f}%")
    print("------")



category_of_interest = "3.1"
category_id_of_interest = next(cat['id'] for cat in coco_data['categories'] if cat['name'] == category_of_interest)

# Instead of a single threshold, we will loop over multiple thresholds to generate the Precision-Recall curve.
thresholds = np.linspace(0, 1, 101)
precisions = []
recalls = []
best_precision = 0
best_threshold = 0
best_recall = 0

for threshold in thresholds:
    # Filter out predictions for the category of interest based on threshold
    thresholded_predictions = [pred for pred in predicted_data if pred["category_id"] == category_id_of_interest and pred["score"] >= threshold]
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for d in coco_data["images"]:
        image_id = d["id"]
        
        # Ground truth boxes for the category of interest
        gt_boxes = [ann["bbox"] for ann in coco_data["annotations"] if ann["image_id"] == image_id and ann["category_id"] == category_id_of_interest]
        pred_boxes = [pred["bbox"] for pred in thresholded_predictions if pred["image_id"] == image_id]

        # If there are no ground truth boxes or predicted boxes for this image, update FP and FN and skip to next iteration
        if not gt_boxes and not pred_boxes:
            continue  # Nothing to do here
        if not gt_boxes:
            false_positives += len(pred_boxes)
            continue
        if not pred_boxes:
            false_negatives += len(gt_boxes)
            continue

        # Convert coco bbox format [x,y,width,height] to [x1,y1,x2,y2]
        gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]
        pred_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in pred_boxes]

        gt_boxes = Boxes(gt_boxes)
        pred_boxes = Boxes(pred_boxes)

        iou_matrix = detectron2.structures.pairwise_iou(gt_boxes, pred_boxes)

        matched_gt = []
        matched_pred = []

        for i, j in enumerate(iou_matrix.argmax(dim=1)):
                if iou_matrix[i, j] >= 0.5:
                    true_positives += 1
                    matched_gt.append(i)
                    matched_pred.append(j)

        false_positives += len(pred_boxes) - len(matched_pred)
        false_negatives += len(gt_boxes) - len(matched_gt)

    # Calculate Precision and Recall
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    if precision > best_precision:
        best_precision = precision
        best_threshold = threshold
        best_recall = recall
    recalls.append(recall)
    precisions.append(precision)

    print("Precision:", precision)
    print("Recall:", recall)

print("the best threshold is ", best_threshold, "with a precision of ", best_precision, "and recall of ", best_recall)
# Plot Precision-Recall curve
plt.plot(recalls, precisions, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve for Category: {category_of_interest}')
plt.grid(True)
plt.show()

# Compute mAP (mean Average Precision)
sorted_indices = np.argsort(recalls)
mAP = np.trapz(np.array(precisions)[sorted_indices], np.array(recalls)[sorted_indices])
print(f"mAP for Category {category_of_interest}: {mAP:.4f}")


