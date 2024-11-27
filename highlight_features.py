import pandas as pd
import cv2
import numpy as np

# Read the CSV file
df = pd.read_csv('provided_data.csv')

# Display the first 5 rows
print(df.head())

# Display basic information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())

# Calculate distance, speed, acceleration, and curvature
def calculate_features(df):
    # Calculate distance traveled
    dx = df.iloc[:, 1].diff().fillna(0)  # x-coordinates
    dy = df.iloc[:, 2].diff().fillna(0)  # y-coordinates
    distance = np.sqrt(dx**2 + dy**2)    # Euclidean distance
    df['Distance Traveled'] = distance.cumsum()  # Cumulative distance
    
    # Calculate speed
    speed = distance  # Assuming frame interval = 1
    df['Speed'] = speed

    # Calculate acceleration (change in speed)
    acceleration = speed.diff().fillna(0)  # Change in speed
    df['Acceleration'] = acceleration

    # Calculate curvature
    # Shift x, y to get three consecutive points
    x_prev, x_next = df.iloc[:, 1].shift(1), df.iloc[:, 1].shift(-1)
    y_prev, y_next = df.iloc[:, 2].shift(1), df.iloc[:, 2].shift(-1)

    # Vector components
    vec1_x, vec1_y = x_prev - df.iloc[:, 1], y_prev - df.iloc[:, 2]
    vec2_x, vec2_y = x_next - df.iloc[:, 1], y_next - df.iloc[:, 2]

    # Calculate dot product and magnitudes
    dot_product = vec1_x * vec2_x + vec1_y * vec2_y
    magnitude1 = np.sqrt(vec1_x**2 + vec1_y**2)
    magnitude2 = np.sqrt(vec2_x**2 + vec2_y**2)

    # Angle (in radians) between the vectors
    angle = np.arccos(np.clip(dot_product / (magnitude1 * magnitude2 + 1e-8), -1, 1))

    # Curvature = angle / distance traveled between points
    curvature = angle / (distance + 1e-8)  # Add small value to prevent division by zero
    df['Curvature'] = curvature.fillna(0)  # Fill NaN values with 0

    return df

# Update DataFrame with new features
df = calculate_features(df)

def create_animation(df):
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('featuresvideo.mp4', fourcc, 30.0, (800, 600))

    # Normalize coordinates to fit within the frame
    x_min, x_max = df.iloc[:, 1].min(), df.iloc[:, 1].max()
    y_min, y_max = df.iloc[:, 2].min(), df.iloc[:, 2].max()
    
    for _, row in df.iterrows():
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Normalize and scale coordinates
        x = int((row.iloc[1] - x_min) / (x_max - x_min) * 780 + 10)
        y = int((row.iloc[2] - y_min) / (y_max - y_min) * 580 + 10)
        
        # Draw the point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {int(row.iloc[0])}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add distance traveled text
        cv2.putText(frame, f"Distance: {row['Distance Traveled']:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add speed to the overlay
        cv2.putText(frame, f"Speed: {row['Speed']:.2f} units/frame", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add acceleration to the overlay
        cv2.putText(frame, f"Accel: {row['Acceleration']:.2f} units/frame^2", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add curvature to the overlay
        cv2.putText(frame, f"Curv: {row['Curvature']:.4f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()

# Create the animation
create_animation(df)

print("Animation saved as 'featuresvideo.mp4'")
