import json
import matplotlib.pyplot as plt

# Read the JSON data
with open('metrics.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Extract iteration, loss, and accuracy values
iterations = [entry['iteration'] for entry in data]
losses = [entry['total_loss'] for entry in data]
accuracies = [entry['fast_rcnn/cls_accuracy'] for entry in data]

# Plot training loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(iterations, losses, color='blue')
plt.title('Training Loss vs. Iteration')
plt.ylim(0,1.4)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(iterations, accuracies, color='green')
plt.title('Training Accuracy vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Training Accuracy')
plt.ylim(0.8,1)
plt.tight_layout()
plt.show()
