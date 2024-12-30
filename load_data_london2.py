from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

print("imported")
def load_and_preprocess(file_paths):
    camera_data = {}

    # Load each CSV and select the relevant columns
    for camera_id, file_path in file_paths.items():
        # Load data
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['day']) + pd.to_timedelta(df['interval'], unit='s')
        #print(df['timestamp'])
        # Extract relevant columns and rename for clarity
        camera_data[camera_id] = df[['timestamp', 'flow']].rename(
            columns={'flow': f'flow_{camera_id}'})
        
        camera_data[camera_id] = camera_data[camera_id].set_index('timestamp')

    # Merge all dataframes on the timestamp index
    merged_df = pd.concat(camera_data.values(), axis=1)
    filled_df = merged_df.copy()

    # Apply the forward and backward fill interpolation to handle NaN values
    filled_df = filled_df.interpolate(method='linear', limit_direction='both')
    for camera_id in file_paths.keys():
        filled_df[f'target_{camera_id}'] = filled_df[f'flow_{camera_id}'].shift(-1)
    filled_df = filled_df.dropna()
    return filled_df

def preprocess_train(df):
    # Initialize the scaler
    overall_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale the entire DataFrame (all columns individually)
    scaled_df = pd.DataFrame(overall_scaler.fit_transform(df), 
                             columns=df.columns, 
                             index=df.index)
    
    return scaled_df, overall_scaler

def preprocess_test(df, sc):
    # Scale the entire DataFrame (all columns individually)
    scaled_df = pd.DataFrame(sc.transform(df),
                             columns=df.columns, 
                             index=df.index)
    
    return scaled_df

def train_test_split(data, mor=4, mid=1, eve=5):
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Extract hour to determine time of day
    data['hour'] = data['timestamp'].dt.hour

    # Define time ranges
    morning = (0, 8)
    mid_day = (8, 16)
    evening = (16, 24)

    # Function to filter data by time range
    def filter_time_range(df, start_hour, end_hour):
        return df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]

    # Create separate test sets for morning, mid-day, and evening from different days
    test_morning = filter_time_range(data, *morning)
    test_mid_day = filter_time_range(data, *mid_day)
    test_evening = filter_time_range(data, *evening)

    # Combine test sets ensuring they are from different days
    # For simplicity, pick one day for each time range
    test_morning_day = test_morning['timestamp'].dt.date.unique()[mor]
    test_mid_day_day = test_mid_day['timestamp'].dt.date.unique()[mid]
    test_evening_day = test_evening['timestamp'].dt.date.unique()[eve]

    test_morning = test_morning[test_morning['timestamp'].dt.date == test_morning_day]
    test_mid_day = test_mid_day[test_mid_day['timestamp'].dt.date == test_mid_day_day]
    test_evening = test_evening[test_evening['timestamp'].dt.date == test_evening_day]

    # Combine test data
    test_data = pd.concat([test_morning, test_mid_day, test_evening]).sort_values('timestamp')
    test_dates = test_data['timestamp'].tolist()
    test_data = test_data.drop(['timestamp', 'hour'], axis=1)
    # Create the remaining data as training data
    train_data = data[~data.index.isin(test_data.index)]
    train_dates = train_data['timestamp'].tolist()
    train_data = train_data.drop(['timestamp', 'hour'], axis=1)

    return train_data, train_dates, test_data, test_dates

def create_sliding_windows(df, dates_df, window_size, feature_columns, target_columns):
    windows = []
    targets = []
    dates = []
    for i in range(len(df) - window_size):
        window = df.iloc[i:i + window_size][feature_columns].values
        target = df.iloc[i + window_size][target_columns].values
        date = dates_df[i + window_size]
        windows.append(window)
        targets.append(target)
        dates.append(date)
    return np.array(windows), np.array(targets), np.array(dates)
