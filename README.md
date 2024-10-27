![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)
# Classification in Machine Learning Project
Classification is a supervised machine learning method where the model tries to predict the 
correct label of a given input data. In classification, the model is fully trained using 
the training data, and then it is evaluated on test data before being used to perform 
prediction on new unseen data.
### Classification Algorithms Evaluated
- Logistic Regression
- Decision Tree
- Random Forests
- K-Nearest Neighbours
- Naive Bayes
### Metrics used for Classification Models Evaluation
**Accuracy:** The proportion of correctly classified instances over the total number 
of instances in the test set.\
**Precision:** Precision measures the proportion of true positives over the total number of predicted positives.\
**Recall:** Recall measures the proportion of true positives over the total number of actual positives.\
**Specificity:** Specificity measures how well the model can correctly identify instances of the negative class.\
**F1-Score:** The harmonic mean of precision and recall, calculated as 2 x (precision x recall) / (precision + 
recall).\
**Confusion matrix:** A table that shows the number of true positives, true negatives, false positives, 
and false negatives for each class, which can be used to calculate various evaluation metrics.\
**Cross-validation:** A technique that divides the data into multiple folds and trains the model on each fold 
while testing on the others, to obtain a more robust estimate of the model’s performance.
### Dataset and it's Features
Can you predict if customer is going to cancel the reservation ?\
A significant number of hotel reservations are called-off due to cancellations or no-shows. The typical reasons 
for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option 
to do so free of charge or preferably at a low cost which is beneficial to hotel guests, but it is a less desirable 
and possibly revenue-diminishing factor for hotels to deal with.\
A [Kaggle Hotel Reservations Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)\ 
is used for the project.\
Dataset has **36275 entries, 19 columns**. One of them (booking_status) is the target column, 12 features columns
with data type of integer or float and 4 features columns with data type of object.\
![Classifier Data Correlation Plot](./img/Figure_data_correlation.png)\
**Dataset is unbalanced** - initial train data class distribution is 19512 for not canceled bookings to 9508 
canceled bookings.
After Train *Dataset Balancing* class distribution is 9508 for not canceled booking to 9508 canceled booking.
## Initial Evaluation results
Classifiers Performance Metrics on Hotel Reservations Dataset are:
![Classifier Performance Metrics Table](./img/Table_1.png)
![Classifier Performance Metrics Figure](./img/Figure_1.png)
## Evaluation results after models tuning
Two models were tuned: Logistic Regression and Random Forest Classifier.
### Logistic Regression
The certain set of a thresholds was used to find the best threshold for the Logistic Regression model: 
![Classifier Performance Metrics Table](./img/Figure_logistic.png)
The best fitted threshold and corresponding accuracy is:
![Classifier Performance Metrics Table](./img/Table_logistic.png)
### Random Forests
The GridSearch Cross Validation method was used to find the best parameters for the Random Forest model:
![Classifier Performance Metrics Plot](./img/Figure_random_forest.png)
## Overfitting Check
![Classifier Overfitting Plot](./img/Figure_lr_overfitting.png)
![Classifier Overfitting Plot](./img/Figure_rf_overfitting.png)
## Data Normalization
Min-Max scaling and Z-score normalization (standardization) are the two fundamental techniques 
for normalization. 
## Features Importance
![Classifier Feature Importance Plot](./img/Figure_features_importance.png)
## Final Results
| Model               | Accuracy          | Overfitting distance |
|---------------------|-------------------|----------------------|
| Logistic Regression | 0.8857            |                      |
| Random Forests      | 0.8928            |                      |
### Project status
Way forward:
- Check for model overfitting.
- Evaluate feature normalisation possibilities.
- Evaluate feature importance.
- Check for data correlation.
- Introduce more params fo Random Forest model.
### Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.
### Resources
[Data Set Link](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)
### License
This project is licensed under the MIT License.
