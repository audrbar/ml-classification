import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV

from data_prep import X_train, X_test, y_train, y_test

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [23, 24, 26],
    'max_depth': [18, 20, 22],
    'bootstrap': [True]
}

# Set up GridSearchCV (Cross Validation)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=6, n_jobs=-1, verbose=2, scoring='accuracy',
                           return_train_score=True)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", grid_search.best_params_)

# Get the best estimator from the grid search
best_rf = grid_search.best_estimator_

# Get the cross-validation results
cv_results = grid_search.cv_results_

# ------------------ Evaluate on Train Data ------------------

# Predict on the training set with the best model
y_train_predict = best_rf.predict(X_train)

# Calculate metrics for the training set
train_accuracy = accuracy_score(y_train, y_train_predict)
train_confusion = confusion_matrix(y_train, y_train_predict)
train_precision = precision_score(y_train, y_train_predict, average='macro')
train_recall = recall_score(y_train, y_train_predict, average='macro')
train_f1 = f1_score(y_train, y_train_predict, average='macro')

# Print training set metrics
print("\n--- Training Set Evaluation ---")
print(f"Train Confusion Matrix: \n{train_confusion}")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Train Precision: {train_precision:.4f}")
print(f"Train Recall: {train_recall:.4f}")
print(f"Train F1 Score: {train_f1:.4f}")

# ------------------ Evaluate on Test Data ------------------

# Predict on the test set with the best model
y_test_predict = best_rf.predict(X_test)

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, y_test_predict)
test_confusion = confusion_matrix(y_test, y_test_predict)
test_precision = precision_score(y_test, y_test_predict, average='macro')
test_recall = recall_score(y_test, y_test_predict, average='macro')
test_f1 = f1_score(y_test, y_test_predict, average='macro')

# Print test set metrics
print("\n--- Test Set Evaluation ---")
print(f"Test Confusion Matrix: \n{test_confusion}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Extract the mean train and test (validation) scores for each estimator
train_scores = cv_results['mean_train_score']
test_scores = cv_results['mean_test_score']
estimators = [f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}"
              for params in cv_results['params']]

# Plot the train and test accuracies for each estimator
plt.figure(figsize=(10, 6))
plt.plot(estimators, train_scores, marker='o', linestyle='-', color='g', label='Train Accuracy')
plt.plot(estimators, test_scores, marker='o', linestyle='-', color='b', label='Test Accuracy')
plt.title('Train and Test Accuracies for Each Estimator')
plt.xlabel('Estimator (n_estimators, max_depth)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
