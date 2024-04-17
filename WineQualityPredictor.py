import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# loading the dataset to a Pandas DataFrame
wine_dataset = pd.read_csv('C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\.idea\\winequality-red.csv')

# number of rows & columns in the dataset
wine_dataset.shape

# first 5 rows of the dataset
wine_dataset.head()

# checking for missing values
wine_dataset.isnull().sum()

# statistical measures of the dataset
wine_dataset.describe()

# creates a categorical plot (count plot) showing the distribution of wine quality levels
sns.catplot(x='quality', data = wine_dataset, kind = 'count')

# creates a bar plot showing the relationship between wine quality and volatile acidity.
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'volatile acidity', data = wine_dataset)

# creates a bar plot showing the relationship between wine quality and citric acid.
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'citric acid', data = wine_dataset)

# calculates the correlation matrix for numerical columns in the dataset
correlation = wine_dataset.corr()

# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')

#  separates the independent variables from the target variable 'quality'
X = wine_dataset.drop('quality',axis=1)
print(X)

# assigns the target variable 'quality' to Y
# wines with quality 7 or higher are labeled as 1 (good quality) and the rest as 0 (bad quality).
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)

# 80% of the data for training and 20% for testing.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y.shape, Y_train.shape, Y_test.shape)

# initializes and fits a Random Forest classifier model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# calculates accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy : ', test_data_accuracy)

# It defines input data for a new wine sample.
input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# It predicts the quality of the new wine sample using the trained Random Forest model.
prediction = model.predict(input_data_reshaped)
print(prediction)

# It prints out whether the predicted quality of the wine is good or bad based on the model's prediction.
if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')