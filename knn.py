import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
# df['Species'] = y

print(df.head())
print(f"{y=}")

plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r']
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, i * 3 + j + 1)
        if i == j:
            for target in range(3):
                plt.hist(X[y == target, i], bins=20, color=colors[target], alpha=0.7, label=iris.target_names[target])
        else:
            for target in range(3):
                plt.scatter(X[y == target, j], X[y == target, i], c=colors[target], label=iris.target_names[target])
        if i == 2:
            plt.xlabel(iris.feature_names[j])
        if j == 0:
            plt.ylabel(iris.feature_names[i])
        plt.legend()
plt.tight_layout()
plt.show()


def custom_standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data


X_normalized = custom_standardize(X)

plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r']
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, i * 3 + j + 1)
        if i == j:
            for target in range(3):
                plt.hist(X_normalized[y == target, i], bins=20, color=colors[target], alpha=0.7,
                         label=iris.target_names[target])
        else:
            for target in range(3):
                plt.scatter(X_normalized[y == target, j], X_normalized[y == target, i], c=colors[target],
                            label=iris.target_names[target])
        if i == 2:
            plt.xlabel(iris.feature_names[j])
        if j == 0:
            plt.ylabel(iris.feature_names[i])
        plt.legend()
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)


def custom_knn(X_train, y_train, X_test, k):
    predictions = []
    for x_test in X_test:
        distances = np.linalg.norm(X_train - x_test, axis=1)
        nearest_neighbors_indices = np.argsort(distances)[:k]
        nearest_neighbors_labels = y_train[nearest_neighbors_indices]
        unique_labels, counts = np.unique(nearest_neighbors_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predictions.append(predicted_label)
    return np.array(predictions)


best_k = None
best_accuracy = 0
for k in range(1, 11):
    y_pred = custom_knn(X_train, y_train, X_test, k)
    accuracy = np.mean(y_pred == y_test)
    # print(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f'Наилучшее значение k: {best_k}')

# y_pred = custom_knn(X_train, y_train, X_test, best_k)


def predict_new_object(X_train, y_train, new_object, k):
    distances = np.linalg.norm(X_train - new_object, axis=1)
    nearest_neighbors_indices = np.argsort(distances)[:k]
    nearest_neighbors_labels = y_train[nearest_neighbors_indices]
    unique_labels, counts = np.unique(nearest_neighbors_labels, return_counts=True)
    predicted_label = unique_labels[np.argmax(counts)]
    return predicted_label


new_object = np.array([5.1, 3.5, 1.4, 0.2])

new_object = (new_object - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

predicted_class = predict_new_object(X_train, y_train, new_object, best_k)

predicted_class_label = iris.target_names[predicted_class]

print(f'Predicted class: {predicted_class_label}')
