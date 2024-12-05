# AutomatedSportsHighlights

This repository showcases my project focused on sports tracking and automating video transitions. The main objectives were to process tracking data, classify player actions, and create smooth video transitions to enhance highlight generation. This project serves as a step in my learning journey, combining machine learning, video processing, and data analysis.

---

## Table of Contents

- [Overview](#overview)
- [Thought Process](#thought-process)
- [Design Choices](#design-choices)
- [Implementation Details](#implementation-details)
- [Installation/Setup](#installationsetup)

---

## Overview

The project involves:
1. **Data Analysis**: Extracting insights and generating visualizations from sports tracking data.
2. **Action Classification**: Training and fine-tuning machine learning models to classify actions based on player movement and effort metrics.
3. **Video Transitions**: Automating video editing with custom transitions to create polished highlight reels.

Key deliverables include a CSV of predictions, a highlight video with transitions, and comprehensive visualizations of derived metrics.

---

## Thought Process

1. **Understanding the Problem**: 
   - The project started by breaking down sports tracking data into manageable components: frame-based tracking, player effort, and movement patterns.
   - The objective was to map this data to meaningful player actions and create an engaging video output.

2. **Feature Engineering**: 
   - Derived features like **distance traveled** and **motion derivatives** were added to enrich the dataset. This helped the model understand nuances in player movements.

3. **Model Optimization**:
   - A two-stage approach was used: **Randomized Search** for broad hyperparameter tuning and **Bayesian Optimization** for fine-tuning.
   - This ensured the model achieved reasonable accuracy despite limited computational resources.

4. **Video Enhancement**: 
   - Transitions such as fades and wipes were implemented to smooth out cuts in highlight reels.
   - Focus was placed on enhancing viewer experience without overcomplicating the process.

5. **Pragmatism Over Perfection**: 
   - Since this was my first time tackling such a project, the emphasis was on functional implementation and learning. Further refinements can be made with more experience.

---

## Design Choices

1. **Gradient Boosting Classifier**:
   - Chosen for its ability to handle structured data efficiently and its interpretability compared to neural networks.

2. **Lag Features for Time-Series**:
   - Added lagged versions of features to capture temporal dependencies, critical for time-series predictions.

3. **Feature Normalization**:
   - StandardScaler was used to ensure consistent scaling, improving model performance and stability.

4. **Video Transitions**:
   - Transition types were chosen for simplicity and relevance: fade, wipe (left/right), and dissolve.

5. **OpenCV for Video Processing**:
   - Selected for its robust library of tools, ensuring high-quality video frame manipulation.

---

## Implementation Details

1. **Data Preprocessing**:
   - Handled missing values in the 'effort' column with linear interpolation.
   - Created derived metrics (e.g., distance traveled) and normalized all features.

2. **Model Development**:
   - Applied a Gradient Boosting Classifier with optimized hyperparameters.
   - Used Randomized Search for initial exploration and Bayesian Optimization for fine-tuning.

3. **Visualization**:
   - Generated subplots for original data and its derivatives to analyze patterns.
   - Saved plots as `.png` files for easy reference.

4. **Video Processing**:
   - Developed custom transitions using OpenCV.
   - Ensured frame-by-frame manipulation for smooth output.

5. **Outputs**:
   - CSV of predictions (`predictions.csv`) aligned with the target format.
   - Highlight video (`output_video.mp4`) showcasing classified actions and transitions.

---
## Installation/Setup

Follow these steps to set up your environment and reproduce the highlight reel from raw data:

### Step 1: Clone the Repository
git clone https://github.com/akeeb2002/AutomatedSportsHighlights.git

cd AutomatedSportsHighlights

### Step 2: Install Anaconda (Optional, Recommended)
Windows:
Download and run the Miniconda installer for Windows: https://docs.conda.io/en/latest/miniconda.html
Open the Anaconda Prompt from your Start menu to proceed with the setup.

macOS:
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
source ~/.bash_profile

### Linux:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

### Step 3: Set Up the Conda Environment
conda env create -f environment.yaml

conda activate mleng_env

### Step 4: Reproduce the Highlight Reel

### Generate additional features and insights from provided_data.csv:
python data_analysis.py
Output: visualizations like plot.png.

python highlight_features.py
Output: featuresvideo.mp4


### Train the model and generate predictions:
python time_classification.py
Input: provided_data.csv, target.csv
Output: predictions.csv

### Smoothing predictions for better results:
python filter_predictions.py
Input: predictions.csv. target.csv
Output: smoothed_predictions.csv, predictions_comparison.png

### Create the final highlight reel:
python opencv_intro.py tracking_visualization.mp4 --csv "smoothed_predictions.csv"

Input: smoothed_predictions.csv
Output: output_video.mp4

### The resulting video can be found in the project directory as output_video.mp4.
