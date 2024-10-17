import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import specificity_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from data_prep import X_train, X_test, y_train, y_test


# 4. Function to train and evaluate classifiers
def evaluate_classifier(model):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='macro')
    recall = recall_score(y_test, y_predict, average='macro')
    specificity = specificity_score(y_test, y_predict, average='macro')
    f1 = f1_score(y_test, y_predict, average='macro')

    return {
        'Classifier': model.__class__.__name__,
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'Specificity': f"{specificity:.4f}",
        'f1': f"{f1:.4f}",
    }


# 3. List of classifiers to evaluate
classifiers = [
    LogisticRegression(solver='liblinear', max_iter=500),
    DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42),
    RandomForestClassifier(n_estimators=18, criterion='entropy', bootstrap=True,
                           max_samples=0.8, random_state=42, max_depth=5),
    KNeighborsClassifier(n_neighbors=8, metric='minkowski'),
    GaussianNB()
]

# 5. Evaluate all classifiers and store the results
accuracy_results = [evaluate_classifier(clf) for clf in classifiers]

# Convert accuracy results to a DataFrame for tabular display
accuracy_df = pd.DataFrame(accuracy_results)
accuracy_df.set_index('Classifier', inplace=True)

# Print the accuracy table
print("\nAccuracy Table for All Classifiers for Hotel Reservations Dataset")
print(accuracy_df)

# Plotting the metrics from accuracy_df as a line chart

# Convert metrics to numeric for plotting
accuracy_df[['Accuracy', 'Precision', 'Recall', 'Specificity', 'f1']] = accuracy_df[
    ['Accuracy', 'Precision', 'Recall', 'Specificity', 'f1']].apply(pd.to_numeric)

# Plot each metric
plt.figure(figsize=(10, 6))
for column in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'f1']:
    plt.plot(accuracy_df.index, accuracy_df[column], marker='o', linestyle='-', label=column)

# Add titles and labels
plt.title('Classifier Performance Metrics on Hotel Reservations Dataset')
plt.xlabel('Classifier')
plt.ylabel('Metric Value')
plt.ylim(0.4, 1)
plt.legend(title='Metrics')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
