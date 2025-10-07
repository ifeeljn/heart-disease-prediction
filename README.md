# heart-disease-prediction
❤️ Heart Disease Prediction Project
A machine learning project focused on predicting heart disease using patient health data. This notebook walks through the data cleaning, exploration, preprocessing, and model training steps, culminating in a Logistic Regression model.

📋 Table of Contents
Dataset Overview

Dependencies and Setup

Data Preprocessing & Cleaning

Exploratory Data Analysis (EDA)

Model Training and Evaluation

Next Steps

💾 Dataset Overview
The dataset contains 1000 entries (patients) across 16 features  related to cardiovascular health.

Feature	Data Type	Non-Null Count	Description (Key Examples)
Age	int64	1000	
Patient age (mean: 52.29 years) 

Cholesterol	int64	1000	
Total Cholesterol level (mean: 249.94) 

Blood Pressure	int64	1000	
Resting Blood Pressure (mean: 135.28) 

Heart Rate	int64	1000	
Resting Heart Rate (mean: 79.20) 

Exercise Hours	int64	1000	
Hours of exercise per week (mean: 4.53) 

Smoking	object	1000	
Smoking status (Current, Never, Former) 


Alcohol Intake	object	660	
Alcohol consumption status 

Heart Disease	int64	1000		
Target Variable (0 = No Disease, 1 = Disease) 


Export to Sheets
🛠️ Dependencies and Setup
This project requires standard data science libraries in Python:

Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
🧹 Data Preprocessing & Cleaning
1. Handling Missing Values
The Alcohol Intake column had missing values (660 non-null out of 1000).

The missing values were attempted to be imputed with the mode of the column.


Note: Due to issues with the fillna operation shown in the notebook , the column was subsequently dropped.


2. Feature Engineering / Encoding

Label Encoding was applied to all categorical features:

Gender

Smoking

Family History

Diabetes

Obesity

Exercise Induced Angina

Chest Pain Type

📈 Exploratory Data Analysis (EDA)
Target Distribution: The Heart Disease target variable is a binary outcome (0 or 1). There is a slight imbalance, with the mean being approximately 0.392 (39.2% of patients have heart disease).


Age and Heart Disease: The average Age for patients with Heart Disease (1) appears notably higher than for those without (0), as visualized in the bar plot.
*

🤖 Model Training and Evaluation
1. Setup

Features (X) and Target (y): The Heart Disease column was dropped to form the feature matrix X , with y as the target vector.






Train-Test Split: Data was split for training and testing. (Note: The test_size=8.2 value in the notebook  appears to be an error/typo, as it should typically be a float between 0.0 and 1.0 or an integer for the absolute number of samples).


Feature Scaling: StandardScaler was applied to standardize the features.

2. Model
A Logistic Regression model was chosen for the binary classification task.

3. Results
Metric	Score
Training Accuracy		
60.875% 


Export to Sheets
Note: A ConvergenceWarning was observed during training, suggesting the model may not have fully converged with default parameters. Increasing max_iter or checking alternative solvers is recommended.


➡️ Next Steps
To improve the model's performance and address the warnings, the following steps are recommended:

Hyperparameter Tuning: Adjust the max_iter in LogisticRegression to achieve convergence.


Evaluate on Test Set: Calculate the accuracy score on x_test and y_test using the generated y_pred.

Alternative Models: Explore other classification algorithms like Random Forest or Support Vector Machines.
