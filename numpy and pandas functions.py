

#computations of numpy functions
#numpy program to convert a list of numerical values into a one dimensional numpy arrays




import numpy as np

def list_to_numpy_array(input_list):
    # Convert the input list to a NumPy array
    numpy_array = np.array(input_list)
    return numpy_array

# Example usage:
input_list = [1, 2, 3, 4, 5]
result_array = list_to_numpy_array(input_list)
print(result_array)


#NumPy program to convert a list and tuple into NumPy arrays:

import numpy as np

def list_and_tuple_to_numpy_arrays(input_list, input_tuple):
    # Convert the input list and tuple to NumPy arrays
    numpy_array_from_list = np.array(input_list)
    numpy_array_from_tuple = np.array(input_tuple)
    return numpy_array_from_list, numpy_array_from_tuple

# Example usage:
input_list = [1, 2, 3, 4, 5]
input_tuple = (10, 20, 30, 40, 50)
result_array_from_list, result_array_from_tuple = list_and_tuple_to_numpy_arrays(input_list, input_tuple)

print("Result array from list:", result_array_from_list)
print("Result array from tuple:", result_array_from_tuple)



#data manipulations using pandas
# import numpy as np
import pandas as pd

# Function to convert a NumPy array to a DataFrame
def numpy_array_to_dataframe(numpy_array):
    df = pd.DataFrame(numpy_array)
    return df

# Function to convert a pandas Series to a DataFrame
def pandas_series_to_dataframe(pandas_series):
    df = pandas_series.to_frame()
    return df

# Example usage:
# Converting a NumPy array to a DataFrame
numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result_df_from_numpy = numpy_array_to_dataframe(numpy_array)
print("DataFrame from NumPy array:")
print(result_df_from_numpy)

# Converting a pandas Series to a DataFrame
pandas_series = pd.Series([10, 20, 30, 40, 50])
result_df_from_series = pandas_series_to_dataframe(pandas_series)
print("\nDataFrame from pandas Series:")
print(result_df_from_series)



#program to add,subtract,multiple and two pandas series
import pandas as pd

# Function to perform addition on two pandas Series
def add_series(series1, series2):
    result = series1 + series2
    return result

# Function to perform subtraction on two pandas Series
def subtract_series(series1, series2):
    result = series1 - series2
    return result

# Function to perform multiplication on two pandas Series
def multiply_series(series1, series2):
    result = series1 * series2
    return result

# Function to perform division on two pandas Series
def divide_series(series1, series2):
    result = series1 / series2
    return result

# Example usage:
series1 = pd.Series([1, 2, 3, 4, 5])
series2 = pd.Series([10, 20, 30, 40, 50])

# Addition
result_addition = add_series(series1, series2)
print("Addition:")
print(result_addition)

# Subtraction
result_subtraction = subtract_series(series1, series2)
print("\nSubtraction:")
print(result_subtraction)

# Multiplication
result_multiplication = multiply_series(series1, series2)
print("\nMultiplication:")
print(result_multiplication)

# Division
result_division = divide_series(series1, series2)
print("\nDivision:")
print(result_division)



#program to retrieve and manipulate data using dataframes
import pandas as pd

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 22, 28, 35],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'Salary': [50000, 60000, 45000, 70000, 55000]
}

df = pd.DataFrame(data)

# Displaying the DataFrame
print("Original DataFrame:")
print(df)

# Retrieving data from DataFrame

# Accessing specific column(s)
print("\nAccessing 'Name' column:")
print(df['Name'])

# Accessing multiple columns
print("\nAccessing 'Name' and 'Age' columns:")
print(df[['Name', 'Age']])

# Accessing specific rows using loc[]
print("\nAccessing row at index 2:")
print(df.loc[2])

# Accessing specific rows using iloc[]
print("\nAccessing row at index 1 using iloc[]:")
print(df.iloc[1])

# Manipulating data in DataFrame

# Adding a new column
df['Gender'] = ['Female', 'Male', 'Male', 'Male', 'Female']
print("\nDataFrame after adding the 'Gender' column:")
print(df)

# Filtering data using conditions
filtered_df = df[df['Age'] > 25]
print("\nFiltered DataFrame (Age > 25):")
print(filtered_df)

# Updating data in a specific row and column
df.at[0, 'Salary'] = 55000
print("\nDataFrame after updating salary of the first row:")
print(df)

# Deleting a column
df.drop('City', axis=1, inplace=True)
print("\nDataFrame after deleting the 'City' column:")
print(df)

# Deleting a row
df.drop(4, inplace=True)
print("\nDataFrame after deleting the row at index 4:")
print(df)

# Sorting the DataFrame based on a column
sorted_df = df.sort_values(by='Age', ascending=False)
print("\nDataFrame sorted by Age in descending order:")
print(sorted_df)



