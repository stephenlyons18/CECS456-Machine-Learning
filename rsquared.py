# Brenden Smith, Konrad Hernandez, Mark Fastner, Stephen Lyons

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Finding R^2 for simple linear regression

# Read in data
dataset = pd.read_csv('Salary_Data.csv')

# Split into X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fit simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test set results
y_pred = regressor.predict(X_test)

# Calculate R^2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R^2:", r2)

# Plot test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('R^2 = ' + str(r2))
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


 



 


 


