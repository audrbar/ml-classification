from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import tree
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from data_prep import X_train, X_test, y_train, y_test, X

# Define hyperparameter grid
param_dict = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    "criterion": ['gini', 'entropy'],
}

# Initialize and train the DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV (Cross Validation) and fit the GridSearchCV to the training data
grid_search = GridSearchCV(model, param_grid=param_dict, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and score from grid search
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Get the best estimator from the grid search
best_dt = grid_search.best_estimator_

# Predict on the test set with the best model
y_predict = best_dt.predict(X_test)

# Visualize Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(best_dt, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title("Decision Tree Classifier Plot")
plt.show()

# Evaluate model performance
confusion = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')

print(f"Confusion Matrix: \n{confusion}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Not_Canceled', 'Canceled'],
            yticklabels=['Not_Canceled', 'Canceled'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature Importance
feature_importance = best_dt.feature_importances_
print("Feature Importance: ", feature_importance)

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Filter out features with importance equal to zero and sort the DataFrame by importance in ascending order
feature_importance_df = feature_importance_df[feature_importance_df['Importance'] != 0]
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Print the sorted feature importance's
print("Sorted Feature Importance's:")
print(feature_importance_df)

# Plot Feature Importance
plt.figure(figsize=(12, 5))
plt.barh(feature_importance_df.Feature, feature_importance_df.Importance, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
