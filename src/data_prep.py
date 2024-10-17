import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load Data and Define the Features and Target
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-classification/data/HotelReservations.csv')
df = df.dropna()

# 1. Display the first few rows of the DataFrame
print(df.info())
# print(pd.Series({c: df[c].unique() for c in df}))
print(df['type_of_meal_plan'].unique())
print(df['room_type_reserved'].unique())
print(df['market_segment_type'].unique())
print(df['booking_status'].unique())
print(f"\nDF Columns: \n{df.columns}")

# 2. Replace the string values with integers using numpy.where
df['type_of_meal_plan'] = (
    np.where(df['type_of_meal_plan'] == 'Not Selected', 0, np.where(df['type_of_meal_plan'] == 'Meal Plan 1', 1,
    np.where(df['type_of_meal_plan'] == 'Meal Plan 2', 2, np.where(df['type_of_meal_plan'] == 'Meal Plan 3', 3,
    df['type_of_meal_plan'])))))
df['room_type_reserved'] = (
    np.where(df['room_type_reserved'] == 'Room_Type 1', 1, np.where(df['room_type_reserved'] == 'Room_Type 2', 2,
    np.where(df['room_type_reserved'] == 'Room_Type 3', 3, np.where(df['room_type_reserved'] == 'Room_Type 4', 4,
    np.where(df['room_type_reserved'] == 'Room_Type 5', 5, np.where(df['room_type_reserved'] == 'Room_Type 6', 6,
    np.where(df['room_type_reserved'] == 'Room_Type 7', 7,  df['room_type_reserved']))))))))
df['market_segment_type'] = (
    np.where(df['market_segment_type'] == 'Offline', 1, np.where(df['market_segment_type'] == 'Online', 2,
    np.where(df['market_segment_type'] == 'Corporate', 3, np.where(df['market_segment_type'] == 'Aviation', 4,
    np.where(df['market_segment_type'] == 'Complementary', 5, df['market_segment_type']))))))
df["booking_status"] = np.where(df["booking_status"] == "Canceled", 0, 1)

X, y = (df[['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
            'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year',
            'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']],
        df['booking_status'])

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
