# Brenden Smith, Stephen Lyons, Mark Fastner, Konrad Hernandez

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Initialize dataset for Support Vector Regression
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(np.reshape(y, (-1, 1)))

# Fit SVR to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predict a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))).reshape(-1, 1))

# Calculate R^2
from sklearn.metrics import r2_score
r2 = r2_score(y, regressor.predict(X))

# Visualize the Support Vector Regression results
# plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color='blue')
# plt.title('R^2 = ' + str(r2))
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Higher resolution & smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('R^2 = ' + str(r2))
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
