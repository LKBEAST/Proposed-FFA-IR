import json

# Load the best.json file
with open("bestmetrics.json", 'r') as file:
    best_metrics = json.load(file)

# Remove 'epoch' and 'train_loss' from best_metrics
best_metrics.pop('epoch', None)
best_metrics.pop('train_loss', None)

best_averages = sum(best_metrics.values()) / len(best_metrics)

# Initialize variable to keep track of the best average score and its filename
best_average_score = best_averages
best_average_filename = "bestmetrics.json"

# Iterate over the xmetrics.json files
for x in range(1, 31):
    filename = f"records/{x}metrics.json"
    
    with open(filename, 'r') as file:
        metrics = json.load(file)
    
    # Remove 'epoch' and 'train_loss'
    metrics.pop('epoch', None)
    metrics.pop('train_loss', None)
    
    # Compare with best metrics
    for key, value in metrics.items():
        if value > best_metrics.get(key, 0):
            print(f"{filename} has a better score in {key}. Score: {value}, Best Score: {best_metrics[key]}")
    
    # Compute average
    average_score = sum(metrics.values()) / len(metrics)
    
    if average_score > best_average_score:
        best_average_score = average_score
        best_average_filename = filename

print(f"\nThe file with the highest average score across all measurements is: {best_average_filename}with an average score of {best_average_score}")

import json
import matplotlib.pyplot as plt

# Load the best.json file
with open("bestmetrics.json", 'r') as file:
    best_metrics = json.load(file)

# Remove 'epoch' and 'train_loss' from best_metrics
best_metrics.pop('epoch', None)
best_metrics.pop('train_loss', None)

# Load 23metrics.json
with open("records/23metrics.json", 'r') as file:
    metrics_23 = json.load(file)

# Remove 'epoch' and 'train_loss' from 23metrics
metrics_23.pop('epoch', None)
metrics_23.pop('train_loss', None)

# Plotting
labels = list(best_metrics.keys())
best_values = list(best_metrics.values())
metrics_23_values = [metrics_23[key] for key in labels]

x = range(len(labels))

plt.figure(figsize=(12, 7))
plt.bar(x, best_values, width=0.4, label='Baseline Model', color='#1f77b4', align='center')
plt.bar(x, metrics_23_values, width=0.4, label='Proposed Model', color='#ff7f0e', align='edge')

plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Performance of the proposed model vs Baseline', fontsize=16)
plt.xticks(x, labels, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



