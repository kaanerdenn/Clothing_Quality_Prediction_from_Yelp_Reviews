# Clothing_Quality_Prediction_from_Yelp_Reviews

This project focuses on predicting clothing quality based on Yelp reviews. It involves data preprocessing, feature engineering, and training a machine learning model to predict clothing quality from various features.

## Dataset

The dataset used in this project is loaded from the "cloth_yelp.json" file, which contains information from Yelp reviews related to clothing. The target variable of interest is "quality."

### Data Exploration

The script begins with data exploration using the `check_df` function to provide an overview of the dataset's shape, data types, missing values, and summary statistics.

## Data Preprocessing

### Removing Irrelevant Columns

Columns with too many null values and those that do not provide significant information, such as "item_id," and "waist," are removed from the dataset.

### Encoding Categorical Data

Categorical columns, including "cup size," "category," "length," "fit," and "shoe width," are encoded using Label Encoding to convert them into numerical format.

### Handling Missing Values

Rows with missing values are removed from the dataset to ensure data quality.

### Extracting Numeric Data

The "bust" and "height" columns are processed to extract numeric values from mixed-format entries. These columns are then cleaned and prepared for analysis.

## Exploratory Data Analysis

The project includes data visualization with seaborn to understand the distribution of "size" and "height_inches" features using box plots. The correlation matrix heatmap is also generated to identify relationships between features.

## Model Training

A Random Forest classifier is used for training the model with the target variable "quality." The dataset is split into training and test sets.

## Model Evaluation

The trained Random Forest model is evaluated using accuracy scores on both the training and test datasets to assess its performance in predicting clothing quality.

## Results

The project provides insights into the process of predicting clothing quality from Yelp reviews using machine learning techniques. It also showcases the data preprocessing and feature engineering steps necessary for building predictive models in this domain.

For more details and specific code implementations, please refer to the project's Python script.
