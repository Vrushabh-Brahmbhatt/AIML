import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names

class NB:
    def fit(self, X, y):
        self.c = np.unique(y)
        self.m = np.array([X[y==c].mean(axis=0) for c in self.c])
        self.v = np.array([X[y==c].var(axis=0) for c in self.c])
        self.p = np.array([np.mean(y==c) for c in self.c])
        
    def _pdf(self, i, x):
        m, v = self.m[i], self.v[i]
        return np.exp(-(x-m)**2/(2*v)) / np.sqrt(2*np.pi*v)
        
    def _pred(self, x):
        post = [np.log(p)+np.sum(np.log(self._pdf(i, x))) for i, p in enumerate(self.p)]
        return self.c[np.argmax(post)]
        
    def predict(self, X):
        return np.array([self._pred(x) for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
nb = NB(); nb.fit(X_train, y_train)
y_pred = nb.predict(X_test
                   
print('Accuracy: %.4f' % np.mean(y_pred==y_test))
print("Predictions:", names[y_pred])
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=names))

# Take user input for a new sample and predict its species
sl = float(input("Enter sepal length: "))
sw = float(input("Enter sepal width: "))
pl = float(input("Enter petal length: "))
pw = float(input("Enter petal width: "))
user_sample = np.array([sl, sw, pl, pw])
pred = nb.predict(np.array([user_sample]))[0]
print("Predicted species:", names[pred])
