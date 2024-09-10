import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, r2_score

# Load the data
datos = pd.read_csv('temperaturas/temperatures.csv')

# Split the data into training and test sets
X = datos['Celsius'].values.reshape(-1,1)
y = datos['Fahrenheit'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Accuracy: {model.score(X_test, y_test)}')
#print(f'R2 Score: {model.score(X_test, y_test)}')

# Plot the results
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.title('Linear Regression Model')
plt.show()

#Predict the temperature in Fahrenheit for a given temperature in Celsius
def predict_temperature(celsius):
    fahrenheit = model.predict(np.array(celsius).reshape(-1,1))
    return fahrenheit[0]

#print(y_test.values)
#print(y_pred)

print(predict_temperature(0))

# Convert predictions and actual values to integers
y_test_rounded = np.round(y_test)
y_pred_rounded = np.round(y_pred)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_rounded, y_pred_rounded)
print('Confusion Matrix:')
print(conf_matrix)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted values')
# plt.ylabel('Actual values')
# plt.title('Confusion Matrix')
# plt.show()


