import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris= load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_predict(train_data, train_labels, test_point, k=3):
    distances = []

    for i in range(len(train_data)):
        distance = euclidean_distance(test_point, train_data[i])
        distances.append((distance, train_labels[i]))

    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_neighbors = sorted_distances[:k]

    class_counts = {}
    for neighbor in k_nearest_neighbors:
        label = neighbor[1]
        class_counts[label] = class_counts.get(label, 0) + 1
    predicted_class = max(class_counts, key=class_counts.get)
    return predicted_class


pred = [knn_predict(X_train, y_train, X_test[i], k=3) for i in range(len(X_test))]
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)


sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

user_data_np = np.array([sepal_length, sepal_width, petal_length, petal_width])

# Reshape the array to have a single row
user_data_np = user_data_np.reshape(1, -1) # reshape to (1, 4) for single sample

prediction = knn_predict(X_train, y_train, user_data_np[0], k=3)

# Map prediction to Iris species
iris_species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

predicted_species = iris_species.get(prediction, "Unknown")
