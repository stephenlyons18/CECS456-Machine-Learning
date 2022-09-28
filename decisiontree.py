# Brenden Smith, Stephen Lyons, Mark Fastner, Konrad Hernandez

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Decision Tree Regression

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fit Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predict a new result
y_pred = regressor.predict(np.array([[6.5]]))

# Visualize the Decision Tree Regression results (higher resolution & smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
