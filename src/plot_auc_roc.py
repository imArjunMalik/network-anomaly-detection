import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import joblib

# Load the saved true labels and predicted probabilities
y_test, y_test_pred_prob = joblib.load('test_predictions.pkl')

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob)

# Calculate AUC
auc = roc_auc_score(y_test, y_test_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.jpeg')  # Save the plot as a JPEG file
plt.show()
