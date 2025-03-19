import sys
#make sure this ridge_regression.py is in the same directory through sys or manual
from ridge_regression import RidgeRegression

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

file_path = '../datasets/Cleaned-Life-Exp.csv'

data = pd.read_csv(file_path)

# print(data.describe)

# data.info()

X = data.drop(['Country', 'Year', 'Life expectancy'], axis=1)
y = data['Life expectancy']


X = X.dropna()  # You can replace this with imputation if necessary

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lambdas = [10**i for i in range(-5, 3)]
train_err = []
test_err = []

from sklearn.metrics import r2_score

for lam in lambdas:
    model = RidgeRegression(lam=lam)
    model.fit(x_train, y_train)
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    train_err.append(mean_squared_error(y_train, y_pred_train))
    test_err.append(mean_squared_error(y_test, y_pred_test)) 

plt.plot(lambdas, train_err, label='train')
plt.plot(lambdas, test_err, label='test')
plt.xlabel('Lambdas')
plt.ylabel('MSE')
plt.legend()
plt.show()
