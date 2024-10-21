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
    'max_depth': [19, 20, 21],
    'bootstrap': [True]
}

# Set up GridSearchCV (Cross Validation)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=6, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_predict = best_rf.predict(X_test)

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
