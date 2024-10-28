import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

# 1. Load and Explore Data
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-classification/data/HotelReservations.csv')
print(df.info())

print("\nUniques:")
print(f"Type of meal plan: {df['type_of_meal_plan'].unique()}")
print(f"Room type reserved: {df['room_type_reserved'].unique()}")
print(f"Market segment type: {df['market_segment_type'].unique()}")
print(f"Booking status: {df['booking_status'].unique()}")

# 2. Handle missing values (if any)
df = df.dropna()  # Or: df = df.fillna(df.mean())
df = df.drop(df.columns[[0]], axis=1)
print('DF Columns: ', df.columns)

# 3. Identify categorical columns (if needed) and apply LabelEncoder:
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# 4. Apply StandardScaler to normalize the numeric features (columns)
print(f"\nDF Describe before normalization: \n{df.describe()}")
# Select numeric columns (int64 and float64)
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Filter numeric columns where the maximum value is greater than 10
filtered_columns = numeric_columns.loc[:, numeric_columns.max() > 10]
print(f"\nDF Filtered Columns: \n{filtered_columns.columns}")

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling only to the filtered columns (those where max > 10)
df[filtered_columns.columns] = scaler.fit_transform(df[filtered_columns.columns])
print(f"\nDF Describe after normalization: \n{df.describe()}")

# 5. Check if there is an imbalance in the classes present in your target variable
initial_target_balance = df['booking_status'].value_counts().reset_index()
print("\nInitial Target Classes Balance: \n", initial_target_balance)

# 6. Set Features (X) and Target (y) using colon syntax
X, y = df.iloc[:, :-1],  df.iloc[:, -1]

# Split dataset into training and test sets (imbalanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Initial y_train class distribution: {Counter(y_train)}")
print(f"Initial y_test class distribution: {Counter(y_test)}")

# Balancing by moving samples (adjusting train-test split)
num_class_0_train = sum(y_train == 0)
num_class_1_train = sum(y_train == 1)

# We want to move some samples from class 1 in y_train to y_test to balance it
num_to_move = num_class_1_train - num_class_0_train

# If there's an imbalance, move the excess samples to the test set
if num_to_move > 0:
    # Get indices of class 1 in the training set
    class_1_indices_train = y_train[y_train == 1].index[:num_to_move]

    # Move these indices from train to test
    X_move = X_train.loc[class_1_indices_train]
    y_move = y_train.loc[class_1_indices_train]

    # Drop these indices from the training set
    X_train = X_train.drop(class_1_indices_train)
    y_train = y_train.drop(class_1_indices_train)

    # Add them to the test set
    X_test = pd.concat([X_test, X_move])
    y_test = pd.concat([y_test, y_move])

# Print the updated distribution
print(f"Updated y_train class distribution: {Counter(y_train)}")
print(f"Updated y_test class distribution: {Counter(y_test)}")

# # Alternative distribution solution
# df_sampled = df.groupby('Class').sample(n=100, random_state=1)

# Calculate the correlation matrix for numeric columns
correlation_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap - Hotel Reservations Data')
plt.show()
