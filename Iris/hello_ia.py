#abrir con pandas mi archivo csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset from a CSV file
df = pd.read_csv('Iris\Iris.csv')
df = df.drop('Id', axis=1)

grupos = df.groupby('Species')
print(grupos)
fig = df[df.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')