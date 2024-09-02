import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics


# Load the Iris dataset from a CSV file and drop the 'Id' column
df = pd.read_csv('./Iris/Iris.csv')
df = df.drop('Id', axis=1)

#print(df.head())
#print(end='\n\n')
#print(df.describe())

# fig = df[df.Species == 'Iris-setosa'].plot(kind='scatter',
#                                                 x='SepalLengthCm',
#                                                 y='SepalWidthCm', 
#                                                 color='red', 
#                                                 label='Setosa')
# df[df.Species == 'Iris-versicolor'].plot(kind='scatter',
#                                             x='SepalLengthCm',
#                                             y='SepalWidthCm',
#                                             color='blue',
#                                             label='Versicolor', ax=fig)
# df[df.Species == 'Iris-virginica'].plot(kind='scatter',
#                                             x='SepalLengthCm',
#                                             y='SepalWidthCm',
#                                             color='green',
#                                             label='Virginica', ax=fig)

# Create a pairplot of the dataset
#sns.pairplot(df, hue='Species')
#plt.savefig('./Iris/pairplot_iris.png')
#plt.show()

# Split the dataset into training and test sets
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=70)
#---------------------K-Nearest Neighbors---------------------#
print('\nK-Nearest Neighbors\n')

# Create a k-NN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Create a confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# Plot a confusion matrix
# heatmap blue to white
sns.heatmap(confusion_matrix, annot=True, cmap='Blues') 
plt.savefig('./Iris/confusion_matrix_iris.png')

#---------------------Regresion Logistica---------------------#
print('\nlogistic regression\n')

# Create a logistic regression classifier
logreg = LogisticRegression(max_iter=200)

# Fit the classifier to the data
logreg.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = logreg.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Create a confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# Plot a confusion matrix
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.savefig('./Iris/confusion_matrix_iris_logistic.png')

#---------------------Support Vector Machine---------------------#
print('\nSupport Vector Machine\n')

# Create a support vector classifier
svm = SVC()

# Fit the classifier to the data
svm.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = svm.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Create a confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# Plot a confusion matrix
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.savefig('./Iris/confusion_matrix_iris_svm.png')

#---------------------Decision Tree---------------------#
print('\nDecision Tree\n')

# Create a decision tree classifier
dt = DecisionTreeClassifier()

# Fit the classifier to the data
dt.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = dt.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Create a confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# Plot a confusion matrix
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.savefig('./Iris/confusion_matrix_iris_dt.png')