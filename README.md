# Milestone-2 

## JupyterNotebook Link
https://colab.research.google.com/drive/1w8f1IqjS9Dk7cO-5kUUaAtw-ux4vd8hW?usp=sharing

## Environment Setup
Must have the following libraries installed:
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Data Preprocessing

In order to preprocess our data, normally we would fill gaps in our data, but because our dataset does not contain any we will skip this step. So, we will first sclae the features in order to make them easier to compare. Then convert the score variable and text into a format that can be easily manipulated by our machine learning model. 

# Sentiment Machine Learning Model

## Where does the model fit in the fitting graph

Test Accuraacy: 1.0
Test Accuracy: 0.575

This model is overffiting the training data, this can be seen by the perfect training accuracy and the test accuracy is significantly lower than that.
```bash
Training Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        79
           4       1.00      1.00      1.00        80

    accuracy                           1.00       159
   macro avg       1.00      1.00      1.00       159
weighted avg       1.00      1.00      1.00       159

Test Classification Report
              precision    recall  f1-score   support

           0       0.59      0.62      0.60        21
           4       0.56      0.53      0.54        19

    accuracy                           0.57        40
   macro avg       0.57      0.57      0.57        40
weighted avg       0.57      0.57      0.57        40
```

## Next Models to Consider 
Support Vector Machine: Can be used to find a boundary between classes with different kernels.
Decision Tree: Can capture non-linear relationships, and can be used to reduce overfitting. 
Neural Network: Can be used for large datasets, and can be used to reduce overfitting.

##Conclusion
The inital logistic regression model has shown to be overfitting. The training accuracy is 1.0 which shows that the model has memorized the data. When this is used towards the test data, it is significantly lower at 0.575, which means that the model does not do well with unseen data. Some improvements that could be made is to implement different features. we could add cross-validation to make sure that the model generalizes well. We could use a SVM machine to try and fight the overfitting and better focus on non-linear relationships. 
