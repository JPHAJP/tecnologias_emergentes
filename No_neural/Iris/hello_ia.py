#abrir con pandas mi archivo csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset from a CSV file
df = pd.read_csv('./Iris/Iris.csv')
df = df.drop('Id', axis=1)

# Create a figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Petal Length vs Petal Width
df[df.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Setosa', ax=ax1)
df[df.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Versicolor', ax=ax1)
df[df.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Virginica', ax=ax1)
ax1.set_xlabel('Petal Length')
ax1.set_ylabel('Petal Width')
ax1.set_title('Petal Length vs Width')

# Plot 2: Sepal Length vs Sepal Width
df[df.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa', ax=ax2)
df[df.Species == 'Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Versicolor', ax=ax2)
df[df.Species == 'Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='red', label='Virginica', ax=ax2)
ax2.set_xlabel('Sepal Length')
ax2.set_ylabel('Sepal Width')
ax2.set_title('Sepal Length vs Width')

# Display the legends
ax1.legend(title='Species')
ax2.legend(title='Species')

# Show the plots
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
