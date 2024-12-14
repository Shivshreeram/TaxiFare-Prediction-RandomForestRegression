# Taxi Fare Prediction using Machine Learning

## Overview

This project illustrates how to use machine learning techniques to predict taxi fares using different features, such as the day of the week, time of day, weather conditions, and traffic conditions. The idea is to create a strong and accurate model that estimates the fare of a taxi ride based on given input factors.

For solving this problem, we used **Random Forest Regression (RFR)**, an advanced ensemble learning algorithm that can capture very complex relationships between the features and the target variable. This approach is best suited for tasks with nonlinear relationships and interactions between the features that characterize the real-world scenario, in this case, taxi fare prediction.

## Problem Description

This project aims to predict taxi fare based on the following features:

- **Day of the Week**: This variable indicates if the trip took place during the weekday or weekend. Weekdays may be busier as they are associated with work, while weekends may be a little erratic as they depend on tourist attractions and recreational activities.

- **Time of Day**: Taxi fares may vary depending on whether the trip happens in the morning, afternoon, evening, or night. For example, taxis during peak hours (morning/evening) may have higher fares due to increased demand.

- **Weather Conditions**: Weather conditions, for example, clear or rainy, can significantly influence both the demand for taxis and the fare itself. Fares may increase when it rains because of increased demand and slow traffic.

- **Traffic Conditions**: Traffic congestion has a direct influence on the time it takes to complete a journey and thus on the fare. More severe traffic conditions (e.g., high congestion) can lead to higher fares due to the longer time spent traveling.

The task is to build a machine learning model that can take these features as input and predict the expected taxi fare.

## Project Workflow

### 1. Data Preprocessing

#### Handling Missing Data:
In any real-world dataset, there are always missing or incomplete entries. This dataset contained missing values, which we handled by filling those missing values with the **median** of each respective column. The median is a robust measure of central tendency that is less sensitive to outliers than the mean.

#### Feature Encoding:
Most of the features in the dataset are categorical variables, such as **Day_of_Week**, **Time_of_Day**, **Weather**, and **Traffic_Conditions**, which the model can only understand if transformed into numbers. Encoding was done in the following manner:

- **Day_of_Week**: 'Weekday' encoded as `1` and 'Weekend' as `2`.
- **Traffic_Conditions**: 'High' was encoded as `2`, 'Medium' as `1`, and 'Low' as `0`.
- **Weather**: 'Clear' was encoded as `1`, and 'Non-Clear' (rainy, cloudy) was encoded as `0`.
- **Time_of_Day**: 'Morning' was encoded with `1`, 'Afternoon' with `2`, 'Evening' with `3`, and 'Night' with `4`.

By this encoding process, the data is transformed into a format that machine learning algorithms can work with, providing meaningful numerical representations of categorical features.

### 2. Model Selection

**Random Forest Regression (RFR)** was the model used to predict taxi fares. RFR is an ensemble learning method that builds multiple decision trees and aggregates their predictions to improve accuracy and reduce overfitting. The key advantages of RFR include:

- **Non-linearity**: RFR can capture non-linear relationships between the features and the target variable, which is essential in a task like taxi fare prediction, as the relationship between the input features and fare is quite complex.
- **Robustness to Overfitting**: Since it constructs many trees, RFR is unlikely to overfit on the training data, which helps the model generalize better to unseen data.
- **Feature Importance**: RFR provides information about which features play the most significant role in predicting the target variable, aiding in better model interpretability.

### 3. Model Training and Hyperparameter Tuning

- **Training**: The model was trained on a subset of the data (training data), and its performance was evaluated on another subset (testing data). This allows us to check how well the model generalizes to new, unseen data.

- **Hyperparameter Tuning**: Random Forest models have several hyperparameters (e.g., the number of trees in the forest and the maximum depth of each tree). Techniques like **Grid Search** were used to identify the best combination of hyperparameters and optimize the model’s performance.

### 4. Model Evaluation

#### Metrics:
We used multiple metrics to evaluate the model's performance:

- **Mean Squared Error (MSE)**: This measures the average of the squared differences between the predicted and actual fare values. The lower the MSE, the better the model's performance.
  
- **Mean Absolute Error (MAE)**: This measures the average of the absolute differences between predicted and actual values. It is an interpretable measure of error in terms of the original scale of the data.

- **Near-Perfect Accuracy**: A custom metric was created to measure how close the predicted values are to actual values within a ±10% tolerance. This provides insight into how well the model performs in real-world scenarios.

### 5. Model Visualization and Explanation

- **Actual vs Predicted Plot**: A scatter plot was developed to plot the predicted values against the actual values. The closer the points are to the ideal line (a 45-degree diagonal line), the better the model's predictions are aligned with reality.

- **Residuals Plot**: A residuals plot was used to check for any patterns in the errors. Ideally, the residuals should scatter around zero with no obvious systematic bias in the model's predictions.

- **Feature Importance**: We used the Random Forest model to determine which features were most influential for predicting taxi fares. The **Traffic Conditions** and **Time of Day** were found to be the most impactful features in predicting fares.

### 6. Conclusion

The project successfully predicted taxi fares with a high degree of accuracy, demonstrating the power of Random Forest Regression in handling complex, non-linear relationships in the data. The model learned from various factors such as time, weather, and traffic conditions to make accurate predictions about taxi fares.

The model, evaluated using metrics like **MSE**, **MAE**, and **Near-Exact Accuracy**, showed good performance and provided reliable and useful predictions for taxi fares.

## About the Data

The dataset used in this project includes information on various factors affecting taxi fares such as the day of the week, time of day, weather conditions, and traffic levels. In the dataset, the `Trip_Price` column initially had missing values (NaN) which represented the taxi fare that needed to be predicted. These missing values were predicted using a Random Forest Regression model, and the resulting predicted values were then assigned back to the `Trip_Price` column of the test data .

## Future Work

While the model performs well with the available features, there are several opportunities for improvement:

- **Adding Other Features**: Incorporating additional features like **distance traveled**, **pickup and drop-off locations**, and **special events** could further enhance the accuracy of the model.
  
- **Exploring Other Models**: While Random Forest performed well, models like **Gradient Boosting**, **XGBoost**, or **Neural Networks** might yield better performance in certain scenarios.

- **Real-Time Fare Prediction**: The next step could be to implement this model into a real-time taxi fare estimation system using live data.

## Dependencies

To run this project, the following libraries are required:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Author

This project was developed by ShivShreeram.

Sophomore ,UG

Date - 14/12/2024

