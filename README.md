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
### Step 2: Set Up a Virtual Environment
# Create a virtual environment
python3 -m venv venv
# Activate the virtual environment
### On macOS/Linux:
source venv/bin/activate
###  On Windows:
venv\Scripts\activate
### Step 3: Install Dependencies
Install the required Python libraries listed in the requirements.txt file.
pip install -r requirements.txt
### Step 4: Organize Input Files
Ensure you have the following files in the appropriate directories:
- Raw Tracking Data: Place your .csv tracking data files in the data/ folder.
- Raw Video Files: Place your corresponding .mp4 video files in the videos/ folder.
You can modify the paths in the scripts if your file organization differs.

## Reproducing the Highlight Reel
Here’s how to go from raw data to a polished highlight reel:




