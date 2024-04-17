import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Importing dataset
DataframeHouse = pd.read_csv('C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\.idea\\USA_Housing.csv')

# Displaying first 5 rows of the dataset
print(DataframeHouse.head())

# Information about the dataframe (number of entries, column data types, and memory usage)
print(DataframeHouse.info())

# Statistical summary of the dataset
print(DataframeHouse.describe())

# Column names
print(DataframeHouse.columns)

# Pairplot for pairwise relationships
sns.pairplot(DataframeHouse)
plt.show()

# Histogram for 'Price' distribution
sns.histplot(DataframeHouse['Price'])
plt.show()

# Heatmap for correlation
sns.heatmap(DataframeHouse.corr(numeric_only=True), annot=True)
plt.show()

# Features and target variable
X = DataframeHouse[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                    'Avg. Area Number of Bedrooms', 'Area Population']]
y = DataframeHouse['Price']

# Splitting the dataset into train and test sets
#  60% of the data for training and 40% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Linear Regression model
lm = LinearRegression()
# Fitting the model
lm.fit(X_train, y_train)

# Intercept and coefficients
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])

# Making predictions on the test set
predictions = lm.predict(X_test)

# Scatter plot for actual vs predicted prices
plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

# Model evaluation
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
