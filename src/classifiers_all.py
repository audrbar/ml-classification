import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import specificity_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load Data and Define the Features and Target
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-classification/data/HotelReservations.csv')
df = df.dropna()

# Display the first few rows of the DataFrame
print(df.info())
# print(pd.Series({c: df[c].unique() for c in df}))
print(df['type_of_meal_plan'].unique())
print(df['room_type_reserved'].unique())
print(df['market_segment_type'].unique())
print(df['booking_status'].unique())
print(f"\nDF Columns: \n{df.columns}")

# Refactor: 'Booking_ID', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
#   'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year',
#   'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
#   'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests', 'booking_status'

# Replace the string values with integers using numpy.where
df['type_of_meal_plan'] = np.where(df['type_of_meal_plan'] == 'Not Selected', 0,
                                   np.where(df['type_of_meal_plan'] == 'Meal Plan 1', 1,
                                            np.where(df['type_of_meal_plan'] == 'Meal Plan 2', 2,
                                                     np.where(df['type_of_meal_plan'] == 'Meal Plan 3', 3,
                                                              df['type_of_meal_plan']))))
df['room_type_reserved'] = np.where(df['room_type_reserved'] == 'Room_Type 1', 1,
                                    np.where(df['room_type_reserved'] == 'Room_Type 2', 2,
                                             np.where(df['room_type_reserved'] == 'Room_Type 3', 3,
                                                      np.where(df['room_type_reserved'] == 'Room_Type 4', 4,
                                                               np.where(df['room_type_reserved'] == 'Room_Type 5', 5,
                                                                        np.where(
                                                                            df['room_type_reserved'] == 'Room_Type 6',
                                                                            7,
                                                                            np.where(df[
                                                                                         'room_type_reserved'] == 'Room_Type 7',
                                                                                     7, df['room_type_reserved'])))))))
df['market_segment_type'] = np.where(df['market_segment_type'] == 'Offline', 1,
                                     np.where(df['market_segment_type'] == 'Online', 2,
                                              np.where(df['market_segment_type'] == 'Corporate', 3,
                                                       np.where(df['market_segment_type'] == 'Aviation', 4,
                                                                np.where(df['market_segment_type'] == 'Complementary',
                                                                         5, df['market_segment_type'])))))
df["booking_status"] = np.where(df["booking_status"] == "Canceled", 0, 1)

X, y = (df[['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
            'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year',
            'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']],
        df['booking_status'])

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to train and evaluate classifiers
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


# List of classifiers to evaluate
classifiers = [
    LogisticRegression(solver='liblinear', max_iter=500),
    DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42),
    RandomForestClassifier(n_estimators=18, criterion='entropy', bootstrap=True,
                           max_samples=0.8, random_state=42, max_depth=5),
    KNeighborsClassifier(n_neighbors=8, metric='minkowski'),
    GaussianNB()
]

# Evaluate all classifiers and store the results
accuracy_results = [evaluate_classifier(clf) for clf in classifiers]

# Convert accuracy results to a DataFrame for tabular display
accuracy_df = pd.DataFrame(accuracy_results)
accuracy_df.set_index('Classifier', inplace=True)

# Print the accuracy table
print("\nAccuracy Table for All Classifiers for Sklearn Breast Cancer Dataset")
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
