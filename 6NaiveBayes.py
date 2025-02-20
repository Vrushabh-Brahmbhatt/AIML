import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.vars = np.array([X[y == c].var(axis=0) for c in self.classes])
        self.priors = np.array([np.sum(y == c) / len(y) for c in self.classes])
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        posteriors = [np.log(prior) + np.sum(np.log(self._pdf(i, x)))
                      for i, prior in enumerate(self.priors)]
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean, var = self.means[class_idx], self.vars[class_idx]
        epsilon = 1e-6  # avoid division by zero
        var += epsilon
        return np.exp(- (x - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train model
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Evaluate model
y_pred = nb.predict(X_test)
print('Accuracy: %.4f' % np.mean(y_pred == y_test))
print("Predictions:", names[y_pred])
print("\nConfusion Matrix:",confusion_matrix(y_test, y_pred))
print("\nClassification Report:",classification_report(y_test, y_pred, target_names=names))

# New sample prediction from user input
sl = float(input("Enter sepal length: "))
sw = float(input("Enter sepal width: "))
pl = float(input("Enter petal length: "))
pw = float(input("Enter petal width: "))
user_data = np.array([sl, sw, pl, pw]).reshape(1, -1)
pred = nb.predict(user_data)[0]
print("Predicted Iris Species:", names[pred])
