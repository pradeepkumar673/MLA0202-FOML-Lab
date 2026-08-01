import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

print("Linear Regression MSE:", mean_squared_error(y, y_lin_pred))
print("Polynomial Regression MSE:", mean_squared_error(y, y_poly_pred))

plt.scatter(X, y, color='black')
plt.plot(X, y_lin_pred, color='blue', label='Linear Fit')
plt.plot(X, y_poly_pred, color='red', label='Polynomial Fit')
plt.legend()
plt.show()
