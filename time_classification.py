import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization

# Load provided_data.csv
data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# Convert 'effort' column to numeric; non-numeric entries will be set to NaN
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# Impute missing 'effort' values using linear interpolation
data['effort'] = data['effort'].interpolate(method='linear')

# Calculate Distance Traveled feature
data['distance_traveled'] = np.sqrt(data['xc'].diff()**2 + data['yc'].diff()**2).fillna(0)

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Load target.csv
target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# Features and target (added 'distance_traveled')
features = ['xc', 'yc', 'w', 'h', 'effort', 'distance_traveled']
X = merged[features]
y = merged['value']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create lag features for time series data
def create_lag_features(X, window_size):
    X_lagged = pd.DataFrame()
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
    return X_lagged.dropna()

window_size = 2  # Define the window size for time series chunks
X_lagged = create_lag_features(X_scaled, window_size)
y_lagged = y.iloc[window_size - 1:]  # Adjust y to align with lagged features
frames_lagged = merged['frame'].iloc[window_size - 1:]  # Get corresponding frame numbers

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# Split into train and test sets (chronological split to respect time series nature)
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]  # Frames corresponding to test set

# Step 1: Randomized Search for Initial Hyperparameter Tuning
param_dist = {
    'n_estimators': [50, 75, 100],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Run RandomizedSearchCV with parallelization
random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5,  # Number of parameter settings to sample
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1  # Parallelization
)
random_search.fit(X_train, y_train)

# Best hyperparameters from RandomizedSearchCV
best_params = random_search.best_params_
print("Best parameters from random search:", best_params)

# Step 2: Bayesian Optimization for Fine-Tuning
# Define a function for Bayesian optimization to optimize
def gb_cv(n_estimators, max_depth, learning_rate, subsample, min_samples_split, min_samples_leaf):
    gb = GradientBoostingClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    gb.fit(X_train, y_train)
    return gb.score(X_test, y_test)

# Set up Bayesian optimizer with the best parameters as a starting point
optimizer = BayesianOptimization(
    f=gb_cv,
    pbounds={
        'n_estimators': (best_params['n_estimators'] - 25, best_params['n_estimators'] + 25),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.2),
        'subsample': (0.5, 1.0),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    },
    random_state=42
)

# Run Bayesian optimization with parallelization
optimizer.maximize(init_points=5, n_iter=5)

# Get the best hyperparameters from Bayesian optimization
optimized_params = optimizer.max['params']
optimized_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] else v 
                    for k, v in optimized_params.items()}  # Adjust data types

# Train final model with optimized hyperparameters
final_clf = GradientBoostingClassifier(
    n_estimators=optimized_params['n_estimators'],
    max_depth=optimized_params['max_depth'],
    learning_rate=optimized_params['learning_rate'],
    subsample=optimized_params['subsample'],
    min_samples_split=optimized_params['min_samples_split'],
    min_samples_leaf=optimized_params['min_samples_leaf'],
    random_state=42
)
final_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = final_clf.predict(X_test)

# Compute and print classification report
print("Classification report with optimized model:")
print(classification_report(y_test, y_pred))

# Write predictions to CSV with the same syntax as target.csv
predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred})
predictions_df.to_csv('predictions.csv', index=False)
