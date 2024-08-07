# Sentiment Machine Learning Model

## JupyterNotebook Link
[https://colab.research.google.com/drive/1w8f1IqjS9Dk7cO-5kUUaAtw-ux4vd8hW?usp=sharing](https://colab.research.google.com/drive/1w8f1IqjS9Dk7cO-5kUUaAtw-ux4vd8hW?usp=sharing)

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

### Imports Needed
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
```
# Final Written Report

## Table of Contents
1. [Introduction to Your Project](#introduction-to-your-project)
2. [Figures](#figures)
3. [Methods Section](#methods-section)
   - [Data Exploration](#data-exploration)
   - [Data Preprocessing](#data-preprocessing)
   - [Model 1: Logistic Regression](#model-1-logistic-regression)
   - [Model 2: Support Vector Machine (SVM)](#model-2-support-vector-machine-svm)
4. [Results Section](#results-section)
   - [Logistic Regression Results](#logistic-regression-results)
   - [SVM Results](#svm-results)
5. [Discussion Section](#discussion-section)
   - [Logistic Regression Discussion](#logistic-regression-discussion)
   - [SVM Discussion](#svm-discussion)
6. [Conclusion](#conclusion)
7. [Statement of Collaboration](#statement-of-collaboration)

## Introduction to Your Project
*Describe the purpose and importance of your project, the dataset used, and the overall goals.*
   In the modern world, social media is a huge and scary thing. Understanding sentiment is crucial for various applications like research, political campaigns, and customer service. Twitter the social media company we know and love, is one of the largest social media platforms. They offered a lot of their data for sentiment analysis. The goal of this project was to develop a machine learning model to classify different tweets based off their sentiment. Sentiment models can automate important task such as, monitoring public post, improving customer service, and maybe most importantly the moderation of content to make social media a safe online space. 
   Twitter recently aquired by Elon Musk, and renamed X resulted in a lot of layoffs at the company. This increased the importance of automating task such as content monitorization which he said was his main cause of employment. A sentiment analysis model like the one I built could help in monitoring content on social media platforms, forums, and gaming chats. Ensuring that online spaces are safe and healthy as we can make them. 
## Figures
![image](https://github.com/user-attachments/assets/d3bff358-f5af-4846-b7f4-b36c2be5fa91)

## Methods Section

### Data Exploration
Using the Sentiment140 Dataset obtained from kaggle. The dataset contained 1.6 million tweets and they were labeled with either positive or negative sentiment. I choose this dataset because it had a good size and was still relevent in todays modern age. The data set includes polarity, tweet Id, data, query, username, and text. 

Steps:
1. Load Data from Kaggle API
2. Inspect Data
```
# Install the Kaggle package
!pip install kaggle

import os
import json

# Move the kaggle.json file to the correct location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Verify if the API is configured correctly
!kaggle datasets list

# Download the Sentiment140 dataset
!kaggle datasets download -d kazanova/sentiment140

# Unzip the dataset
!unzip sentiment140.zip -d sentiment140

# Load the dataset into a pandas DataFrame
file_path = 'sentiment140/training.1600000.processed.noemoticon.csv'
DATA_COL = ['score', 'id', 'date', 'flag', 'user', 'text']

#Read File
data = pd.read_csv(file_path, names=DATA_COL, encoding='latin-1')
#Prints data info
data.info()
```
### Data Preprocessing
This involved cleaning the data and its text. Then preparing the data in order to be test

Steps:
1. Text Cleaning: Remove usernames, URLS, special characters, and stopwords
2. Text Vectorization: converted text into a TF-IDF
3. Data Splitting: split data into training and testing sets

```
#Further Set up Data
data = data[['text', 'score']]

# Cleaning the Data
text = data[['text']]
score = data[['score']]

pos_score = data[data['score'] > 0]
neg_score = data[data['score'] < 0]

dataset = pd.concat([pos_score, neg_score])

stopwords = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it",
    "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ]

def clean_data(text):
  text = text.str.lower()
  # Convert to lowercase
  text = text.str.lower()
  #Remove Usernames
  text = text.apply(lambda x: re.sub(r'@\w+', '', x))
  # Remove special characters and numbers
  text = text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
  # Remove extra spaces
  text = text.apply(lambda x: re.sub(r'\s+', ' ', x))
  # Remove stopwords
  text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
  return text

# Apply the cleaning function to the 'text' column
data['text'] = clean_data(data['text'])
X = data['text']
y = data['score'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the cleaned text
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# Scale the features
#scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)

# Encode the target variable
y = data['score']
```

### Model 1: Logistic Regression
A logistic regression model that was trained and evaluated initially showed signs of overfitting. When used with the 1.6 tweet size model and evaluated the results showed that the model was well-fitted with around 75% accuracy for both training and testing  datasets. 

```
# Create a logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
```

### Model 2: Support Vector Machine (SVM)
A svm model was chosen for its ability to handle non-linear relationships. Originally with the smaller model it was overfitted with a training accuracy of around 0.625. As of now this model with the 1.6 million tweets is still running. Will run it overnight and update if a result in concluded. Im pretty sure Alan Turing said something like "It works, its just... still working" haha 

```
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

#Prediction
y_train_pred_svm = svm_model.predict(X_train)
y_test_pred_svm = svm_model.predict(X_test)

#Accuracy
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
```

## Results Section 

### Logistic Regression Results

- Train Accuracy: 0.77697890625
- Test Accuracy: 0.77406875

Train Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.76      0.77    640506
           4       0.77      0.80      0.78    639494

    accuracy                           0.78   1280000
   macro avg       0.78      0.78      0.78   1280000
weighted avg       0.78      0.78      0.78   1280000

Test Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.75      0.77    159494
           4       0.76      0.80      0.78    160506

    accuracy                           0.77    320000
   macro avg       0.77      0.77      0.77    320000
weighted avg       0.77      0.77      0.77    320000

Train Confusion Matrix:
[[483897 156609]
 [128858 510636]]
Test Confusion Matrix:
[[119968  39526]
 [ 32772 127734]]

### SVM Results

SVM Train Accuracy: 1.0
SVM Test Accuracy: 0.625

SVM Train Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        79
           4       1.00      1.00      1.00        80

    accuracy                           1.00       159
   macro avg       1.00      1.00      1.00       159
weighted avg       1.00      1.00      1.00       159

SVM Test Classification Report:
              precision    recall  f1-score   support

           0       0.71      0.48      0.57        21
           4       0.58      0.79      0.67        19

    accuracy                           0.62        40
   macro avg       0.65      0.63      0.62        40
weighted avg       0.65      0.62      0.62        40

SVM Train Confusion Matrix:
[[79  0]
 [ 0 80]]
SVM Test Confusion Matrix:
[[10 11]
 [ 4 15]]


## Discussion Section

### Logistic Regression Model
This was chosen as the initial model because of its simplicity. It provided a good baseline to compare against my other model. The logistic regression model was training on the 1.6 million tweet dataset. Initially with a smaller version it was overfitted, however, once the full dataset was used, the model balanced its performance to about 77% accuracy on both training and testing datasets. The logistic regression model was able to generalize well however, i wish we couldve achieved a higher accuracy. I believe my results are believable and are close to my expectations. Logistic regression performs well with enough data, and test and training accuracies show that the model generalized correclty. Some shortcomings, may have came from not being able to calculate non-linear relationships in the data.

### SVM Model
This was chosen for its ability to handle non-linear relationships. SVM's are able to prevent overfitting which my initial smaller models suffered with. The SVM model when trained on the smaller subset also showed overfitting, and struggled to generalize data. The SVM was trained on the 1.6 million tweet dataset and has been extremly time-consuming run-time wise. I left it to run overnight and will update if any success is made. I believe that my initial data was overfitting which is common when there is not enough data. With more data the SVM will generalize better. Hopefully the running results will prove me correct. Some shortcomings would be the computational expense of the SVM, this can be extremly long and it may still not be accurate or well-fitted. 

## Conclusion
This project allowed me to go through developing a sentiment machine learning model. The initial model of logistic regression was a good starting point due to its simplicity. It allowed for a basis to compare to other models. Within the full dataset it was accurate and well-fitted. With the addition of a SVM model, we did this to try and fight against overfitting and deal with non-linear relationships. The expectation is that on the larger model the SVM will also do better. 

Some future directions i would go to are using more advanced models like neural networks or Naive Bayes models. We could potentially add some hypertuning like grid or random search in order to enhance model performance. This wouldve potentially aided in the running of the SVM. The last thing i could think of is trying to find a real time way to run this on social media platforms. This would make it a lot more usable in modern day worl and could be used as a bot to monitor sentiment on social media platforms. 

Overall, Each step provided me with new insights, this led me to make better decisions regarding selecting a model and preproccessing techniques. I was happy with the performance of the logistic regression model as it had a good performance and was well-fitted. The SVM has showed me that i need to put more importance into fine tuning and being careful with computational resources.

## Statement of Collaboration
Antonio Lopez: Bossman: Data exploration, preprocessing, model training, evaluation, and report writing.
