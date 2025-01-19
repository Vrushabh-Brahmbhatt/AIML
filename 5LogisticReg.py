import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib

# Fixing the Matplotlib backend issue
matplotlib.use('Agg')  # 'Agg' is a non-GUI backend suitable for saving plots to files

# Load the diabetes dataset from sklearn
diabetes = load_diabetes()
print(diabetes.feature_names)

# The features (X) and the target (y)
X, y = diabetes.data, diabetes.target
print(X[0], y[0])
print(f"Median {np.median(y)}")

# Convert the target variable to binary (1 for diabetes, 0 for no diabetes)
y_binary = (y > np.median(y)).astype(int)
print(y_binary[0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Standardize features: Scale the features to have mean 0 and variance 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision boundary with accuracy information (using BMI and Age)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 8], hue=y_test, palette={0: 'blue', 1: 'red'}, marker='o')
plt.xlabel("BMI")  # Body Mass Index (feature 2)
plt.ylabel("Age")  # Age (feature 8)
plt.title(f"Logistic Regression Decision Boundary\nAccuracy: {accuracy * 100:.2f}%")
plt.legend(title="Diabetes", loc="upper right")

# Ensure the 'plots' directory exists before saving the plot
if not os.path.exists('plots'):
    os.makedirs('plots')

# Save the plot as a PNG file in the 'plots' directory
plt.savefig('plots/decision_boundary.png')  # Save decision boundary plot as a PNG file
plt.close()  # Close the plot to free up memory

# Plot the ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve\nAccuracy: {accuracy * 100:.2f}%')
plt.legend(loc="lower right")

# Save the ROC curve plot as a PNG file in the 'plots' directory
plt.savefig('plots/roc_curve.png')  # Save ROC curve plot as a PNG file
plt.close()  # Close the plot to free up memory

