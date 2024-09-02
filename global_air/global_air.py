#abrir con pandas mi archivo csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#pip install

# Load the Iris dataset from a CSV file
df = pd.read_csv('global_air/global_air.csv')

# Display the first few rows
print(df.head())

# Display the last few rows
#print(df.tail())

# Display the shape of the dataset
print(df.shape)

# Display the columns
print(df.columns)

# Display the data types
print(df.dtypes)

# Display the summary statistics
#print(df.describe())

# Display the number of missing values for each column
print(df.isnull().sum())

grupos = df.groupby('Country').size()
print(grupos)

#fig = df[df.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')