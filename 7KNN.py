import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names

# Split data (25% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def knn_predict(X_train, y_train, x, k=3):
    dists = [euclidean_distance(x, xi) for xi in X_train]
    k_idx = np.argsort(dists)[:k]
    return Counter(y_train[k_idx]).most_common(1)[0][0]

# Evaluate on test set
y_pred = np.array([knn_predict(X_train, y_train, x, k=3) for x in X_test])
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Predictions:", names[y_pred])

# Print confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=names))

# User input for a new prediction
sl = float(input("Enter sepal length: "))
sw = float(input("Enter sepal width: "))
pl = float(input("Enter petal length: "))
pw = float(input("Enter petal width: "))
user_sample = np.array([sl, sw, pl, pw])
pred = knn_predict(X_train, y_train, user_sample, k=3)
print("Predicted species:", names[pred])
