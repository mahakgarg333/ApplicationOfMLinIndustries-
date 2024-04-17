import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Obtain Air Quality Data
data = pd.read_csv('C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\.idea\\station_day.csv')

# Step 2: Clean the Dataset
# Handle missing values, outliers, and inconsistencies
data.dropna(inplace=True)  # Drop rows with missing values
# Handle outliers if necessary

# Step 3: Explore the Dataset
# Descriptive statistics
print(data.describe())
# Visualizations
sns.pairplot(data)
plt.show()

# Step 4: Identify Relevant Features
# Assuming features like pollutant levels, weather conditions, and geographical information are present
# Here, we're considering PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene as features

# Step 5: Engineer New Features
# No new features added in this example

# Step 6: Split the Dataset
X = data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]  # Features
y = data['AQI']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Choose Machine Learning Models
# Assuming Linear Regression and Random Forest Regression
linear_reg_model = LinearRegression()
rf_model = RandomForestRegressor()

# Step 8: Train the Models
linear_reg_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Step 9: Evaluate Model Performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)

print("Linear Regression Model Performance:")
evaluate_model(linear_reg_model, X_test, y_test)
print("\nRandom Forest Model Performance:")
evaluate_model(rf_model, X_test, y_test)

# Step 10: Visualize Predictions
# Example: Scatter plot for Linear Regression model
plt.scatter(y_test, linear_reg_model.predict(X_test))
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Linear Regression: Actual vs. Predicted AQI")
plt.show()

# Step 11: Interpret Trained Models
# Example: Coefficients for Linear Regression model
print("Linear Regression Coefficients:")
print(linear_reg_model.coef_)

# Step 12: Visualize Feature Importances
# Example: Feature importance for Random Forest model
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Random Forest: Feature Importances")
plt.show()