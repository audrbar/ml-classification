from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from data_prep import X_train, X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=500)
model.fit(X_train_scaled, y_train)

# Predictions and probabilities for the test set
y_test_probability = model.predict_proba(X_test_scaled)[:, 1]

# Predictions and probabilities for the training set
y_train_probability = model.predict_proba(X_train_scaled)[:, 1]

# Set custom threshold
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

# Store accuracy for each threshold
accuracies_test = []
accuracies_train = []

# Iterate over different thresholds
for threshold in thresholds:
    # Generate test predictions based on the current threshold
    y_test_custom_th = [1 if prob >= threshold else 0 for prob in y_test_probability]

    # Generate train predictions based on the current threshold
    y_train_custom_th = [1 if prob >= threshold else 0 for prob in y_train_probability]

    # Calculate test accuracy for the current threshold
    accuracy_test = accuracy_score(y_test, y_test_custom_th)
    accuracies_test.append(accuracy_test)

    # Calculate train accuracy for the current threshold
    accuracy_train = accuracy_score(y_train, y_train_custom_th)
    accuracies_train.append(accuracy_train)

    # Print accuracy for each threshold
    print(f"Threshold: {threshold}, Accuracy_test: {accuracy_test:.4f}, Accuracy_train: {accuracy_train:.4f}")

# Plot the accuracies for each threshold
plt.figure(figsize=(8, 4))
plt.plot(thresholds, accuracies_test, marker='o', linestyle='-', color='b', label='Test Accuracy')
plt.plot(thresholds, accuracies_train, marker='o', linestyle='-', color='r', label='Train Accuracy')
plt.title('Accuracy vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()
