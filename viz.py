import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np


def organize_data():
    """
    Load, organize, and normalize data
    """
    df = pd.read_csv('data/2017.csv')
    df.drop(['Arena', 'Rk', 'L', 'PL', 'PW', 'W', 'SOS', 'SRS', 'ORtg', 'DRtg', 'Attendance'], 1, inplace=True)
    df = df.set_index('Team')
    df = (df - df.mean()) / df.std()
    y = df['MOV']
    X = df.drop('MOV', 1)
    return X, y

X, y = organize_data()

def add_param(X, y, current_params):
    scores = []
    errors = []
    columns = X.columns
    if len(current_params) == 0:
        current_estimate = np.mean(y)
    else:
        model = LinearRegression()
        model.fit(X[current_params], y)
        current_estimate = model.predict(X[current_params])
    for column in columns:
        model = LinearRegression()
        test_params = current_params + [column]
        model.fit(X[test_params], y)
        score = model.score(X[test_params], y)
        scores.append(score)
        error = model.predict(X[test_params])
        errors.append(error)
        


add_param(X, y, [])
