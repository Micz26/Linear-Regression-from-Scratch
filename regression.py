import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack(([1] * X.shape[0], X))
        arr = np.linalg.inv(X.T @ X) @ (X.T @ y)
        if self.fit_intercept:
            self.intercept = arr[0]
            self.coefficient = arr[1:]
        else:
            self.coefficient = arr

    def predict(self, X):
        return self.intercept + X @ self.coefficient

    def r2_score(self, y, yhat):
        u = np.sum((y - yhat) ** 2)
        l = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / l

    def rmse(self, y, yhat):
        mse = np.sum((y - yhat) ** 2)
        return np.sqrt(mse / len(y))


def main():
    data = {'f1': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
            'f2': [65.2, 78.9, 61.1, 45.8, 54.3, 58.7, 96.1, 100.0, 85.9, 84.3],
            "f3": [25.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2],
            "y": [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]}

    df = pd.DataFrame(data)
    X = np.array(df[["f1", "f2", "f3"]])
    y = np.array(df["y"].values)

    regCustom = CustomLinearRegression()
    regCustom.fit(X, y)

    regSklearn = LinearRegression()
    regSklearn.fit(X, y)

    custom_dict = {
        'Intercept': regCustom.intercept,
        'Coefficient': regCustom.coefficient,
        'R2': regCustom.r2_score(y, regCustom.predict(X)),
        'RMSE': regCustom.rmse(y, regCustom.predict(X))
    }
    print(custom_dict)

    sklearn_dict = {
        'Intercept': regSklearn.intercept_,
        'Coefficient': regSklearn.coef_,
        'R2': r2_score(y, regSklearn.predict(X)),
        'RMSE': np.sqrt(mean_squared_error(y, regSklearn.predict(X)))
    }
    print(sklearn_dict)

    compare_dict = {
        'Intercept': regSklearn.intercept_ - regCustom.intercept,
        'Coefficient': regSklearn.coef_ - regCustom.coefficient,
        'R2': r2_score(y, regSklearn.predict(X)) - regCustom.r2_score(y, regCustom.predict(X)),
        'RMSE': np.sqrt(mean_squared_error(y, regSklearn.predict(X))) - regCustom.rmse(y, regCustom.predict(X))
    }
    print(compare_dict)

if __name__ == "__main__":
    main()
