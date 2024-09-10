import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the Breast Cancer dataset from a CSV file
data = pd.read_csv('cancer/Cancer_Data.csv')

# Remove the 'Unnamed: 32' column
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

#Display dtype and non-null values of the features
print(data.info())

# Encode the diagnosis column
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Display the first five rows of the data
print(data.head())


# Split the data into features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Logistic Regression Accuracy:', accuracy)

# Display the confusion matrix with heatmap
print('Confusion Matrix:')
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Display the classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Display the correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix)
plt.title('Correlation Matrix')
plt.show()


# K-Nearest Neighbors
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('K-Nearest Neighbors Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Support Vector Machine
model_svm = SVC()
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Support Vector Machine Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Decision Tree
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred = model_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Decision Tree Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Plot the accuracy of the models
models = ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree']
accuracies = [model_lr.score(X_test, y_test), model_knn.score(X_test, y_test), model_svm.score(X_test, y_test), model_dt.score(X_test, y_test)]

plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

# Save the model with the highest accuracy
models = [model_lr, model_knn, model_svm, model_dt]
accuracies = [model_lr.score(X_test, y_test), model_knn.score(X_test, y_test), model_svm.score(X_test, y_test), model_dt.score(X_test, y_test)]
best_model = models[np.argmax(accuracies)]
print('Best Model:', best_model)

