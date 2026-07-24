import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('ch2_vancouver_housing_price.csv')
print('The first 5 rows of the CSV dataset:')
print(data.head())

# Scatter plots for Price vs Each Feature.
fig1 = plt.subplot(2, 3, 1)
plt.scatter(data['Avg. Area Income'], data['Price'])
plt.title('Price VS Average Income')

fig2 = plt.subplot(2, 3, 2)
plt.scatter(data['Avg. Area House Age'], data['Price'])
plt.title('Price VS Average House Age')

fig3 = plt.subplot(2, 3, 3)
plt.scatter(data['Avg. Area Number of Rooms'], data['Price'])
plt.title('Price VS Average Number of Rooms')

fig4 = plt.subplot(2, 3, 4)
plt.scatter(data['Area Population'], data['Price'])
plt.title('Price VS Area Population')

fig5 = plt.subplot(2, 3, 5)
plt.scatter(data['size'], data['Price'])
plt.title('Price VS Size')

#plt.show()

X = data['size']
y = data['Price']
print('\nThe first 5 rows of the Price (y) column:')
print(y.head())

# Reshape X for model training.
X = np.array(X).reshape(-1, 1)
print('\nThe shape of the X array:')
print(X.shape)

# Set up the linear regression model.
from sklearn.linear_model import LinearRegression
LR1 = LinearRegression()

# Train the model.
LR1.fit(X, y)

# Calculate the predicted prices
y_predict_1 = LR1.predict(X)

# Print the first 5 house size, house price, and predicted prices to check the result.
print('\nThe first 5 rows of the X array:')
print(X[0:5])
print('\nThe first 5 rows of the y array:')
print(y[0:5])
print('\nThe first 5 rows of the y_predict_1 array:')
print(y_predict_1[0:5])

# Generate a plot of the actual prices and the predicted prices.
fig6 = plt.figure(figsize=(8, 5))
plt.scatter(X, y)
plt.plot(X, y_predict_1, 'r')
#plt.show()

# Calculate the R-squared value to evaluate the model's performance.
from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error_1 = mean_squared_error(y, y_predict_1)
r2_score_1 = r2_score(y, y_predict_1)
print("The mean squared error (single feature) is", mean_squared_error_1, " and the r2 score is", r2_score_1)

# Define X_multi, create an array that contains all features except Price.
X_multi = data.drop('Price', axis=1)
# Set up 2nd linear model
LR_multi = LinearRegression()
# Train the model.
LR_multi.fit(X_multi, y)
# Make prediction
y_predict_multi = LR_multi.predict(X_multi)
# Generate a new plot of the actual prices and the predicted prices.
fig7 = plt.figure(figsize=(8, 5))
plt.scatter(y, y_predict_multi)
#plt.show()

# Evaluate the model's performance.
mean_squared_error_multi = mean_squared_error(y, y_predict_multi)
r2_score_multi = r2_score(y, y_predict_multi)
print("The mean squared error (multiple features) is", mean_squared_error_multi, " and the r2 score is", r2_score_multi)
