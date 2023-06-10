import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from cnn_model import CNN

# Define the paths to the test data and the trained model
TEST_DATA_PATH = "test/"
MODEL_PATH = "cnn.pkl"

# Define the transformation for test images
transform_img = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the test dataset
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform_img)
test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)

# Create an instance of the CNN model
cnn = CNN()

# Load the trained model
cnn.load_state_dict(torch.load(MODEL_PATH))

# Evaluate the model
cnn.eval()  # Change model to 'eval' mode

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Print the evaluation metrics
print('Accuracy: %.6f' % accuracy)
print('Precision: %.6f' % precision)
print('Recall: %.6f' % recall)
print('F1-Score: %.6f' % f1)