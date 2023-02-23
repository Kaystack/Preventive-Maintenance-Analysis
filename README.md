# Preventive-Maintenance-Analysis 

Preventive maintenance analysis is a technique used to identify potential equipment failures and prevent them from occurring by regularly conducting maintenance activities. The analysis involves reviewing maintenance records and analyzing the data to identify patterns of equipment failure and determine the root causes of the failures. This information is then used to develop a preventive maintenance plan that outlines the specific maintenance activities that need to be performed and the frequency with which they should be conducted to prevent equipment failure. The goal of preventive maintenance analysis is to improve equipment reliability, minimize downtime, and reduce maintenance costs by identifying and addressing potential issues before they become major problems. This technique is commonly used in industries such as manufacturing, transportation, and energy to ensure that equipment is operating efficiently and effectively.

# Installation
The project requires installation of the following libraries:

- NumPy
- Pandas
- Scikit-learn (Sklearn)
- Matplotlib
- Seaborn
- Imbalanced-learn (Imblearn)

To install the libraries, run the following command:
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn

# Usage
To use the project, download the preventive_maintenance.csv file and the Preventive_Maintenance_KNN_Analysis.ipynb notebook. Open the notebook in Jupyter Notebook or JupyterLab and run the code cells.

# Dataset
The preventive maintenance dataset used in this analysis contains information on the maintenance of machines, including the date of the last maintenance and whether the machine experienced a failure since the last maintenance. The dataset has 10000 rows and 4 columns. We are interested in predicting whether a machine will fail or not based on the other variables.

# KNN Analysis
The KNN algorithm is used to train the model to predict whether a machine will fail or not. The dataset is split into a training set and a test set, and KNN is fitted on the training set. The accuracy of the model is calculated using the test set.

The code in the Preventive_Maintenance_KNN_Analysis.ipynb notebook performs the following steps:

- Import necessary libraries
- Load the preventive_maintenance.csv dataset
- Explore the dataset to view first 5 rows and 5 last rows
- Get information about the dataset to understand it, including the shape and identifying the class
- Select the independent variables X and dependent variable y
- Use seaborn to view class label distribution to determine if class balancing is required
- Split the dataset into test set and training set
- Balance the class using RandomOverSampler
- Fit the KNN into the training set
- Predict the dataset
- Calculate the accuracy of the model
- Display the confusion matrix
- Display the classification report
- Visualize the confusion matrix using seaborn heatmap

# Results
The KNN model achieved an accuracy of 90%. The confusion matrix shows that the model correctly predicted 2321 machine failures and 561 non-failures, and incorrectly predicted 109 non-failures and 9 failures.

The classification report shows precision, recall, and F1-score for each class.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# DESCRIPTION:
This project analyzes preventive maintenance data using the K-Nearest Neighbors (KNN) algorithm. The preventive maintenance dataset used in this analysis contains information on the maintenance of machines, including the date of the last maintenance and whether the machine experienced a failure since the last maintenance. The dataset has 10000 rows and 4 columns. The project aims to predict whether a machine will fail or not based on the other variables.

The KNN algorithm is used to train the model to predict whether a machine will fail or not. The dataset is split into a training set and a test set, and KNN is fitted on the training set. The accuracy of the model is calculated using the test set.

The code in the Preventive_Maintenance_KNN_Analysis.ipynb notebook performs the following steps:

- Import necessary libraries
- Load the preventive_maintenance.csv dataset
- Explore the dataset to view first 5 rows and 5 last rows
- Get information about the dataset to understand it, including the shape and identifying the class
- Select the independent variables X and dependent variable y
- Use seaborn


