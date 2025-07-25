import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(f"Model Accuracy: {accuracy_score(y_test, clf.predict(X_test)) * 100:.2f}%")

new_sample = np.array([X_test[0]])
print(f"Predicted Class for the new sample: {'Benign' if clf.predict(new_sample) == 0 else 'Malignant'}")

plt.figure(figsize=(19,8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()