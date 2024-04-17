import pandas as pd

dataset_path = 'C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\company_employee_details.csv'

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\company_employee_details.csv')
# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Handle missing values (replace them or drop them based on your analysis)

# Get a summary of the dataset
print("\nSummary of the dataset:")
print(df.describe())

# Select a subset of columns using label-based indexing
selected_columns_label = df[['company', 'department', 'age', 'salary']]

# Select a subset of columns using position-based indexing
selected_columns_position = df.iloc[:, [0, 1, 3, 6]]  # Assuming columns 0, 1, 3, and 6 are of interest

# Creating a new DataFrame by filtering rows based on a condition
condition = df['years_in_the_company'] > 5  # Example condition: Employees with more than 5 years in the company
filtered_df = df[condition]
# Display the selected columns and the filtered DataFrame
print("\nSelected columns using label-based indexing:")
print(selected_columns_label)

print("\nSelected columns using position-based indexing:")
print(selected_columns_position)

print("\nFiltered DataFrame based on a condition (more than 5 years in the company):")
print(filtered_df)

# Identify missing values
missing_values = df.isnull().sum()

# Display columns with missing values and their counts
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Decide on a strategy to handle missing values (e.g., imputation or removal)

# Impute missing values with the mean for numeric columns
df['annual_bonus'].fillna(df['annual_bonus'].mean(), inplace=True)

# Remove rows with missing values in other columns
df.dropna(subset=['prior_years_experience'], inplace=True)

# Display the updated DataFrame after handling missing values
print("\nDataFrame after handling missing values:")
print(df)

# Create a new column by applying a mathematical operation
df['total_compensation'] = df['salary'] + df['annual_bonus']

# Convert a categorical variable into numerical representation using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['department'], prefix='dept', drop_first=True)

# Display the DataFrame with the new column and one-hot encoding
print("\nDataFrame with new column and one-hot encoding:")
print(df_encoded)

# Group the data by the 'company' column
grouped_data = df.groupby('company')

# Apply aggregation functions (sum, mean, count) to the grouped data
aggregated_data = grouped_data.agg({'salary': 'sum', 'annual_bonus': 'mean', 'age': 'count'})

# Rename the columns for better presentation
aggregated_data = aggregated_data.rename(
    columns={'salary': 'total_salary', 'annual_bonus': 'average_annual_bonus', 'age': 'employee_count'})

# Display the aggregated results
print("\nAggregated data by company:")
print(aggregated_data)

import pandas as pd

# Create the first dataset
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 22, 28]
})

# Create the second dataset
df2 = pd.DataFrame({
    'ID': [2, 3, 4, 5],
    'Salary': [50000, 60000, 70000, 80000],
    'Department': ['HR', 'IT', 'Finance', 'Marketing']
})

# Display the first few rows of each dataset
print("First few rows of df1:")
print(df1)

print("\nFirst few rows of df2:")
print(df2)

# Inner Join
inner_merged = pd.merge(df1, df2, on='ID', how='inner')
print("\nInner Merged Dataset:")
print(inner_merged)
# Left Join
left_merged = pd.merge(df1, df2, on='ID', how='left')
print("\nLeft Merged Dataset:")
print(left_merged)

# Right Join
right_merged = pd.merge(df1, df2, on='ID', how='right')
print("\nRight Merged Dataset:")
print(right_merged)

# Outer Join
outer_merged = pd.merge(df1, df2, on='ID', how='outer')
print("\nOuter Merged Dataset:")
print(outer_merged)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\company_employee_details.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# I. Create a bar plot, line plot, and scatter plot using Pandas plotting functions
# Bar Plot
df['company'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Employees by Company')
plt.xlabel('Company')
plt.ylabel('Number of Employees')
plt.show()

# Line Plot
df['age'].plot(kind='line', color='orange', marker='o')
plt.title('Age Distribution of Employees')
plt.xlabel('Employee Index')
plt.ylabel('Age')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\LENOVO\\PycharmProjects\\pythonProject\\company_employee_details.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# I. Create a bar plot, line plot, and scatter plot using Pandas plotting functions
# Bar Plot
df['company'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Employees by Company')
plt.xlabel('Company')
plt.ylabel('Number of Employees')
plt.show()

# Line Plot
df['age'].plot(kind='line', color='orange', marker='o')
plt.title('Age Distribution of Employees')
plt.xlabel('Employee Index')
plt.ylabel('Age')
plt.show()

import numpy as np

# 5.1: Create NumPy array 'arr' with values from 1 to 10
arr = np.arange(1, 11)

# 5.2: Create another NumPy array 'arr2' with values from 11 to 20
arr2 = np.arange(11, 21)

# 5.3: Perform operations and print the results
addition_result = arr + arr2
subtraction_result = arr - arr2
multiplication_result = arr * arr2
division_result = arr / arr2

print("Array 'arr':", arr)
print("Array 'arr2':", arr2)

print("Addition Result:", addition_result)
print("Subtraction Result:", subtraction_result)
print("Multiplication Result:", multiplication_result)
print("Division Result:", division_result)

# 6.1: Reshape 'arr' into a 2x5 matrix [rows x columns]
arr_2x5 = arr.reshape(2, 5)

# 6.2: Transpose the matrix obtained in the previous step [5x2]
transposed_matrix = arr_2x5.T

# 6.3: Flatten the transposed matrix into a 1D array
flattened_array = transposed_matrix.flatten()

# 6.4: Stack 'arr' and 'arr2' vertically
stacked_result = np.vstack((arr, arr2))

# Print the results
print("\nReshaped 'arr' (2x5 matrix):")
print(arr_2x5)

print("\nTransposed Matrix:")
print(transposed_matrix)

print("\nFlattened Array:")
print(flattened_array)

print("\nVertically Stacked Result:")
print(stacked_result)

# 7.1: Calculate the mean, median, and standard deviation of 'arr'
mean_value = np.mean(arr)
median_value = np.median(arr)
std_deviation_value = np.std(arr)

# 7.2: Find the maximum and minimum values in 'arr'
max_value = np.max(arr)
min_value = np.min(arr)

# 7.3: Normalize 'arr' (subtract the mean and divide by the standard deviation)
normalized_arr = (arr - mean_value) / std_deviation_value

# Print the results
print("\nMean of 'arr':", mean_value)
print("Median of 'arr':", median_value)
print("Standard Deviation of 'arr':", std_deviation_value)
print("Maximum value in 'arr':", max_value)
print("Minimum value in 'arr':", min_value)

print("\nNormalized 'arr':", normalized_arr)

# 8.1: Create a boolean array 'bool_arr' for elements in 'arr' greater than 5
bool_arr = arr > 5

# 8.2: Use 'bool_arr' to extract the elements from 'arr' that are greater than 5
filtered_elements = arr[bool_arr]

# Print the results
print("\nBoolean array for elements greater than 5:")
print(bool_arr)
print("\nElements in 'arr' greater than 5:")
print(filtered_elements)

# 9.1: Generate a 3x3 matrix with random values between 0 and 1
random_matrix = np.random.rand(3, 3)

# 9.2: Create an array of 10 random integers between 1 and 100
random_integers = np.random.randint(1, 101, 10)

# 9.3: Shuffle the elements of 'arr' randomly
np.random.shuffle(arr)

# Print the results
print("\nRandom 3x3 Matrix:")
print(random_matrix)

print("\nArray of 10 Random Integers between 1 and 100:")
print(random_integers)

print("\nShuffled 'arr':", arr)

# 10.1: Apply the square root function to all elements in 'arr'
sqrt_arr = np.sqrt(arr)

# 10.2: Use the exponential function to calculate ex for each element in 'arr'
exp_arr = np.exp(arr)

# Print the results
print("\nSquare root of 'arr':", sqrt_arr)
print("\nExponential function applied to 'arr':", exp_arr)

# 11.1: Create a 3x3 matrix 'mat_a' with random values
mat_a = np.random.rand(3, 3)

# 11.2: Create a 3x1 matrix 'vec_b' with random values
vec_b = np.random.rand(3, 1)

# 11.3: Multiply 'mat_a' and 'vec_b' using the dot product
result = np.dot(mat_a, vec_b)

# Print the results
print("\nMatrix 'mat_a':")
print(mat_a)

print("\nMatrix 'vec_b':")
print(vec_b)

print("\nResult of mat_a * vec_b (dot product):")
print(result)

# 12.1 Create a 2D array 'matrix' with values from 1 to 9
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 12.2 Calculate the mean of each row
row_means = np.mean(matrix, axis=1, keepdims=True)

# Subtract the mean of each row from each element in that row
normalized_matrix = matrix - row_means

print("\nOriginal Matrix:")
print(matrix)

print("\nMatrix after subtracting row means:")
print(normalized_matrix)
