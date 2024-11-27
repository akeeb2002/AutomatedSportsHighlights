import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


@dataclass
class SmoothingParams:
    window_size: int
    min_duration: int
    hysteresis: float
    frame_expansion: int  # Added this for post-smoothing expansion
    
    def to_dict(self) -> dict:
        return {
            'window_size': self.window_size,
            'min_duration': self.min_duration,
            'hysteresis': self.hysteresis,
            'frame_expansion': self.frame_expansion
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SmoothingParams':
        return cls(
            window_size=d['window_size'],
            min_duration=d['min_duration'],
            hysteresis=d['hysteresis'],
            frame_expansion=d.get('frame_expansion', 0)
        )

class PredictionSmoother:
    """
    A class that smooths frame-by-frame predictions and can optimize its parameters
    to match target sequences.
    """
    
    def __init__(self):
        # Default parameter search spaces
        self.default_param_grid = {
            'window_size': [3, 5, 7, 9, 11],
            'min_duration': [1, 2, 3, 4, 5, 10, 30, 60, 120],
            'hysteresis': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            'frame_expansion': [0, 5, 10, 15]  # Added for frame expansion
        }
    
    def smooth_predictions(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        """
        Smooth frame-by-frame predictions using specified parameters and expand active windows.
        """
        if not predictions:
            return []
        
        # Convert input predictions to integers, just in case
        predictions = [int(pred) for pred in predictions]
        
        # Ensure window_size is odd
        window_size = max(3, params.window_size if params.window_size % 2 == 1 
                         else params.window_size + 1)
        half_window = window_size // 2
        
        # Step 1: Apply sliding window majority voting
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            
            active_ratio = sum(window) / len(window)
            
            threshold = 0.5
            if smoothed:
                threshold = params.hysteresis if smoothed[-1] else 1 - params.hysteresis
                
            smoothed.append(1 if active_ratio >= threshold else 0)
        
        # Step 2: Remove short duration state changes
        if params.min_duration > 1:
            final = []
            current_state = smoothed[0]
            current_duration = 1
            
            for pred in smoothed[1:]:
                if pred == current_state:
                    current_duration += 1
                else:
                    if current_duration >= params.min_duration:
                        final.extend([current_state] * current_duration)
                    else:
                        final.extend([not current_state] * current_duration)
                    current_state = pred
                    current_duration = 1
            
            # Handle the last sequence
            if current_duration >= params.min_duration:
                final.extend([current_state] * current_duration)
            else:
                final.extend([not current_state] * current_duration)
                
            smoothed = final

        # Step 3: Expand active windows
        if params.frame_expansion > 0:
            expanded = smoothed[:]
            for i in range(len(smoothed)):
                if smoothed[i] == 1:
                    start = max(0, i - params.frame_expansion)
                    end = min(len(smoothed), i + params.frame_expansion + 1)
                    for j in range(start, end):
                        expanded[j] = 1
            smoothed = expanded
        
        return [int(pred) for pred in smoothed]
    
    def optimize_parameters(self, raw_predictions: List[int], target_sequence: List[int], metric: str = 'f1_score'):
        """
        Optimize smoothing parameters to maximize a specified evaluation metric.

        Args:
            raw_predictions: List of raw predictions (0 or 1)
            target_sequence: List of target values (0 or 1)
            metric: Evaluation metric to optimize ('f1_score' or 'accuracy')

        Returns:
            best_params: Dictionary of best parameters found
            best_metrics: Dictionary containing all the relevant metrics
        """
        best_params = None
        best_metric_value = -1
        best_metrics = {}

        for window_size in self.default_param_grid['window_size']:
            for min_duration in self.default_param_grid['min_duration']:
                for hysteresis in self.default_param_grid['hysteresis']:
                    for frame_expansion in self.default_param_grid['frame_expansion']:
                        params = SmoothingParams(
                            window_size=window_size,
                            min_duration=min_duration,
                            hysteresis=hysteresis,
                            frame_expansion=frame_expansion
                        )
                        smoothed_predictions = self.smooth_predictions(raw_predictions, params)

                        # Compute the chosen metric
                        f1 = f1_score(target_sequence, smoothed_predictions)
                        accuracy = accuracy_score(target_sequence, smoothed_predictions)
                        precision = precision_score(target_sequence, smoothed_predictions)
                        recall = recall_score(target_sequence, smoothed_predictions)
                        
                        # Calculate state change difference (this is an example; adjust based on your requirements)
                        state_change_difference = sum([1 for i in range(1, len(smoothed_predictions)) if smoothed_predictions[i] != smoothed_predictions[i-1]])

                        # Store the metrics
                        metrics = {
                            "accuracy": accuracy,
                            "f1_score": f1,
                            "precision": precision,
                            "recall": recall,
                            "state_change_difference": state_change_difference
                        }

                        # Select the best parameters based on the chosen metric
                        metric_value = metrics[metric]

                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            best_params = params
                            best_metrics = metrics

        return best_params, best_metrics


    def save_params(self, params: SmoothingParams, filepath: str):
        """Save parameters to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(params.to_dict(), f, indent=2)
    
    def load_params(self, filepath: str) -> SmoothingParams:
        """Load parameters from a JSON file."""
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        return SmoothingParams.from_dict(params_dict)



def plot_predictions(raw_predictions: List[int], smoothed_predictions: List[int], target_sequence: List[int], filepath: str):
    """
    Create a plot comparing raw predictions, smoothed predictions, and target sequence.
    
    Args:
        raw_predictions: List of raw predictions (0 or 1)
        smoothed_predictions: List of smoothed predictions (0 or 1)
        target_sequence: List of target values (0 or 1)
        filepath: Path to save the plot
    """
    plt.figure(figsize=(15, 6))
    x = range(len(raw_predictions))
    
    plt.step(x, raw_predictions, where='post', label='Raw Predictions', alpha=0.7)
    plt.step(x, smoothed_predictions, where='post', label='Smoothed Predictions', alpha=0.7)
    plt.step(x, target_sequence, where='post', label='Target Sequence', alpha=0.7)
    
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.title('Comparison of Raw and Smoothed Predictions')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Read CSV files
    predictions_df = pd.read_csv('predictions.csv')
    target_df = pd.read_csv('target.csv')

    # Merge dataframes on 'frame' column and sort by frame
    merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')

    # Convert values to integers (0 and 1)
    raw_predictions = merged_df['value_pred'].astype(int).tolist()
    target_sequence = merged_df['value_target'].astype(int).tolist()

    # Create smoother and optimize parameters
    smoother = PredictionSmoother()
    best_params, best_metrics = smoother.optimize_parameters(
        raw_predictions,
        target_sequence,
        metric='f1_score'  # Could also use 'accuracy' or other metrics
    )

    # Apply smoothing with best parameters
    smoothed_predictions = smoother.smooth_predictions(raw_predictions, best_params)

    # Print results
    print("\nBest parameters found:")
    print(json.dumps(best_params.to_dict(), indent=2))
    print("\nMetrics with best parameters:")
    print(json.dumps(best_metrics, indent=2))

    # Create a new dataframe with smoothed predictions
    result_df = pd.DataFrame({
        'frame': merged_df['frame'],
        'value': [int(pred) for pred in smoothed_predictions]  # Explicitly convert to int
    })

    # Write the result to a new CSV file
    result_df.to_csv('smoothed_predictions.csv', index=False)

    print("\nSmoothed predictions have been written to 'smoothed_predictions.csv'")

    # Create and save the plot
    plot_predictions(raw_predictions, smoothed_predictions, target_sequence, 'predictions_comparison.png')
    print("\nPredictions comparison plot has been saved to 'predictions_comparison.png'")

    # Example of saving and loading parameters
    smoother.save_params(best_params, "best_smoothing_params.json")
    loaded_params = smoother.load_params("best_smoothing_params.json")
