import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

def iterative_forward_select(X, y, depth=None):
    """
    Performes iterative forward selection on X with targets y.
    Currently, only model is linear regression, but this should be
    simple to generalize.

    Args:
        X (pd.DataFrame): DataFrame of predictor variables, dimensions (n_samples, n_features)
        y (pd.Series): Series of target values, dimensions (n_samples, 1)
        depth (int): Number of iterations before terminating

    Returns:
        None
        Outputs visualization of each iteration.  For each forward select iteration,
        the residuals of the previous model are plotted against each predictor variable.  
        Each plot is ranked by how predictive the predictor variable is after 
        already regressing on the previously selected variables.
    """
    if not depth:
        depth = len(X.columns)
    current_params = []
    iteration = 1
    while iteration < depth:
        best_feature = forward_select(X, y, current_params, iteration)
        current_params.append(best_feature)
        iteration += 1
    return None

def forward_select(X, y, current_params, iteration):
    """
    A single round of forward selection.  Determines most predictive variable, 
    and plots how predictive each variable after regressing on previously selected
    variables.

    Args:
        X (pd.DataFrame): DataFrame of predictor variables, dimensions (n_samples, n_features)
        y (pd.Series): Series of target values, dimensions (n_samples, 1)
        current_params (list): list of variables already selected for regression
        iteration (int): which iteration through forward selection are we on

    Returns:
        Selected Variable (str): Name of variable that minimizes training error. 
            Note, this is the best variable after already regressing on previously
            selected variables
    """
    scores = []
    columns = np.array([x for x in X.columns if x not in current_params])
    for column in columns:
        model = LinearRegression()
        test_params = current_params + [column]
        model.fit(X[test_params], y)
        score = model.score(X[test_params], y)
        scores.append(score)
    columns = columns[np.argsort(scores)][::-1]
    scores = np.sort(scores)[::-1]
    residuals = get_residuals(X, y, current_params)
    show_correlations(X, residuals, current_params, iteration, scores, columns)
    return columns[0]

def get_residuals(X, y, current_params):
    """
    Calculates residuals for current model

    Args:
        X (pd.DataFrame): DataFrame of predictor variables, dimensions (n_samples, n_features)
        y (pd.Series): Series of target values, dimensions (n_samples, 1)
        current_params (list): list of variables already selected for regression
        iteration (int): which iteration through forward selection are we on

    Returns:
        residuals (list): list of residuals for each observation
    """
    if len(current_params) == 0:
        current_estimate = np.mean(y)
        residuals = y - current_estimate
    else:
        model = LinearRegression()
        model.fit(X[current_params], y)
        current_estimate = model.predict(X[current_params])
        residuals = y - current_estimate
    return residuals

def show_correlations(X, residuals, current_params, iteration, scores, columns):
    """
    Makes plot of correlation between each remaining predictor variable and residuals.
    Each plot is ranked by how predictive the predictor variable is after 
    already regressing on the previously selected variables.

    Args:
        X (pd.DataFrame): DataFrame of predictor variables, dimensions (n_samples, n_features)
        residuals (list): list of residuals for each observation
        current_params (list): list of variables already selected for regression
        iteration (int): which iteration through forward selection are we on
        scores (list): list of scores representing how well a variable improves the previous model
        columns (list): list of predictor variable names

    Returns: 
        None
        Outputs files in plots/ directory with filenames being the predictor 
        variable names included in the previous model.
    """
    n_features = len(columns)
    f, axarr = plt.subplots(n_features, sharex=True, sharey=True,
                            figsize=(6, 6 * n_features))
    for i in range(n_features):
        axarr[i].scatter(X[columns[i]], residuals)
        axarr[i].set_ylabel(residuals.name, fontsize=18)
        axarr[i].set_xlabel(columns[i], fontsize=18)
        axarr[i].set_title('R^2: ' + str(scores[i]), fontsize=18)
    plt.tight_layout()
    filename = str(iteration) + '_' + '_'.join(current_params)
    filename = filename.replace("/", "")
    filename = filename.replace(".", "")
    plt.savefig('plots/' + filename)
    return None

