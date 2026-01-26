# Run using Python 3.11
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import xgboost
import graphviz
from IPython.display import display, Image
from sklearn.model_selection import KFold
import statsmodels.api as sm
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create baseline DAG
names = [
    "Bugs reported",
    "Monthly usage",
    "Sales calls",
    "Economy",
    "Discount",
    "Last upgrade",
    "Ad spend",
    "Interactions",
]
g = graphviz.Digraph(format="png")
for name in names:
    g.node(name, fontsize="10")
g.node("Product need", style="dashed", fontsize="10")
g.node("Bugs faced", style="dashed", fontsize="10")
g.node("Did renew", style="filled", fontsize="10")

g.edge("Product need", "Did renew")
g.edge("Product need", "Discount")
g.edge("Product need", "Bugs reported")
g.edge("Product need", "Monthly usage")
g.edge("Discount", "Did renew")
g.edge("Monthly usage", "Bugs faced")
g.edge("Monthly usage", "Did renew")
g.edge("Monthly usage", "Ad spend")
g.edge("Economy", "Did renew")
g.edge("Sales calls", "Did renew")
g.edge("Sales calls", "Product need")
g.edge("Sales calls", "Interactions")
g.edge("Interactions", "Did renew")
g.edge("Bugs faced", "Did renew")
g.edge("Bugs faced", "Bugs reported")
g.edge("Last upgrade", "Did renew")
g.edge("Last upgrade", "Ad spend")

out_file = os.path.join(script_dir, "renewal_graph1")
out_path = g.render(filename=out_file, cleanup=True)

# DAG for impact of ad spend on renewal
names = [
    "Bugs reported",
    "Monthly usage",
    "Sales calls",
    "Economy",
    "Discount",
    "Last upgrade",
    "Ad spend",
    "Interactions",
]

g = graphviz.Digraph(format="png")  # Set output format to PNG

for name in names:
    g.node(name, fontsize="10")

g.node("Product need", style="dashed", fontsize="10")
g.node("Bugs faced", style="dashed", fontsize="10")
g.node("Did renew", style="filled", fontsize="10")

g.edge("Product need", "Did renew")
g.edge("Product need", "Discount")
g.edge("Product need", "Bugs reported")
g.edge("Product need", "Monthly usage")
g.edge("Discount", "Did renew")
g.edge("Monthly usage", "Bugs faced")
g.edge("Monthly usage", "Did renew", color="blue")
g.edge("Monthly usage", "Ad spend", color="blue")
g.edge("Economy", "Did renew")
g.edge("Sales calls", "Did renew")
g.edge("Sales calls", "Product need")
g.edge("Sales calls", "Interactions")
g.edge("Interactions", "Did renew")
g.edge("Bugs faced", "Did renew")
g.edge("Bugs faced", "Bugs reported")
g.edge("Last upgrade", "Did renew", color="blue")
g.edge("Last upgrade", "Ad spend", color="blue")

out_file = os.path.join(script_dir, "renewal_adspend")
out_path = g.render(filename=out_file, cleanup=True)

# DAG of discount on renewal
names = [
    "Bugs reported",
    "Monthly usage",
    "Sales calls",
    "Economy",
    "Discount",
    "Last upgrade",
    "Ad spend",
    "Interactions",
]

g = graphviz.Digraph(format="png")  # Set output format to PNG

for name in names:
    g.node(name, fontsize="10")

g.node("Product need", style="dashed", fontsize="10")
g.node("Bugs faced", style="dashed", fontsize="10")
g.node("Did renew", style="filled", fontsize="10")

g.edge("Product need", "Did renew", color="red")
g.edge("Product need", "Discount", color="red")
g.edge("Product need", "Bugs reported")
g.edge("Product need", "Monthly usage", color="blue")
g.edge("Discount", "Did renew", style="dashed", color="blue")
g.edge("Monthly usage", "Bugs faced")
g.edge("Monthly usage", "Did renew", color="blue")
g.edge("Monthly usage", "Ad spend", color="blue")
g.edge("Economy", "Did renew")
g.edge("Sales calls", "Did renew")
g.edge("Sales calls", "Product need")
g.edge("Sales calls", "Interactions")
g.edge("Interactions", "Did renew")
g.edge("Bugs faced", "Did renew")
g.edge("Bugs faced", "Bugs reported")
g.edge("Last upgrade", "Did renew", color="blue")
g.edge("Last upgrade", "Ad spend", color="blue")

out_file = os.path.join(script_dir, "renewal_discount")
out_path = g.render(filename=out_file, cleanup=True)

# Estimate effect of Ad Spend on Renewal using boosted trees
# This is just for illustration, no tuning and using simulated data

# Define helper functions
class FixableDataFrame(pd.DataFrame):
    """Helper class for manipulating generative models."""

    def __init__(self, *args, fixed={}, **kwargs):
        self.__dict__["__fixed_var_dictionary"] = fixed
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        out = super().__setitem__(key, value)
        if isinstance(key, str) and key in self.__dict__["__fixed_var_dictionary"]:
            out = super().__setitem__(key, self.__dict__["__fixed_var_dictionary"][key])
        return out


# generate the data
def generator(n, fixed={}, seed=0):
    """The generative model for our subscriber retention example."""
    if seed is not None:
        np.random.seed(seed)
    X = FixableDataFrame(fixed=fixed)

    # the number of sales calls made to this customer
    X["Sales calls"] = np.random.uniform(0, 4, size=(n,)).round()

    # the total number of interactions with the customer
    X["Interactions"] = X["Sales calls"] + np.random.poisson(0.2, size=(n,))

    # the health of the regional economy this customer is a part of
    X["Economy"] = np.random.uniform(0, 1, size=(n,))

    # the time since the last product upgrade when this customer came up for renewal
    X["Last upgrade"] = np.random.uniform(0, 20, size=(n,))

    # how much the user perceives that they need the product
    X["Product need"] = X["Sales calls"] * 0.1 + np.random.normal(0, 1, size=(n,))

    # the fractional discount offered to this customer upon renewal
    X["Discount"] = ((1 - scipy.special.expit(X["Product need"])) * 0.5 + 0.5 * np.random.uniform(0, 1, size=(n,))) / 2

    # What percent of the days in the last period was the user actively using the product
    X["Monthly usage"] = scipy.special.expit(X["Product need"] * 0.3 + np.random.normal(0, 1, size=(n,)))

    # how much ad money we spent per user targeted at this user (or a group this user is in)
    X["Ad spend"] = (
        X["Monthly usage"] * np.random.uniform(0.99, 0.9, size=(n,)) + (X["Last upgrade"] < 1) + (X["Last upgrade"] < 2)
    )

    # how many bugs did this user encounter in the since their last renewal
    X["Bugs faced"] = np.array([np.random.poisson(v * 2) for v in X["Monthly usage"]])

    # how many bugs did the user report?
    X["Bugs reported"] = (X["Bugs faced"] * scipy.special.expit(X["Product need"])).round()

    # did the user renew?
    X["Did renew"] = scipy.special.expit(
        7
        * (
            0.18 * X["Product need"]
            + 0.08 * X["Monthly usage"]
            + 0.1 * X["Economy"]
            + 0.05 * X["Discount"]
            + 0.05 * np.random.normal(0, 1, size=(n,))
            + 0.05 * (1 - X["Bugs faced"] / 20)
            + 0.005 * X["Sales calls"]
            + 0.015 * X["Interactions"]
            + 0.1 / (X["Last upgrade"] / 4 + 0.25)
            + X["Ad spend"] * 0.0
            - 0.45
        )
    )

    X["Did renew"] = scipy.stats.bernoulli.rvs(X["Did renew"])

    return X


def user_retention_dataset():
    """The observed data for model training."""
    n = 10000
    X_full = generator(n)
    y = X_full["Did renew"]
    X = X_full.drop(["Did renew", "Product need", "Bugs faced"], axis=1)
    return X, y


def fit_xgboost(X, y, loss="logistic"):
    """Train an XGBoost model with early stopping."""
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    params = {
        "eta": 0.001,
        "subsample": 0.5,
        "max_depth": 2,
    }

    if loss == "logistic":
        params["objective"] = "reg:logistic"
    elif loss == "squared_error":
        params["objective"] = "reg:squarederror"
    else:
        raise ValueError("Invalid loss function. Choose 'logistic' or 'squared_error'.")

    model = xgboost.train(
        params,
        dtrain,
        num_boost_round=200000,
        evals=((dtest, "test"),),
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    return model

# Generate data
X, y = user_retention_dataset()

Ad = X["Ad spend"]
Xc = X.drop(["Ad spend"], axis=1)

# Need to make folds just like for cross-validation
cv = KFold(n_splits=5, shuffle = True, random_state = 220)

# Store residuals
residuals_y = []
residuals_Ad = []

# Loop through folds
for train_idx, test_idx in cv.split(Xc):
    # Split data
    X_train, X_test = Xc.iloc[train_idx], Xc.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    Ad_train, Ad_test = Ad.iloc[train_idx], Ad.iloc[test_idx]

    # Train models
    model_y = fit_xgboost(X_train, y_train, loss="logistic")
    model_Ad = fit_xgboost(X_train, Ad_train, loss="squared_error")

    # Predict on the test set
    y_pred = model_y.predict(xgboost.DMatrix(X_test))  # Predicted probability
    Ad_pred = model_Ad.predict(xgboost.DMatrix(X_test))

    # Compute residuals
    residuals_y.extend(y_test - y_pred)
    residuals_Ad.extend(Ad_test - Ad_pred)

# Convert residuals to NumPy array
residuals_y = np.array(residuals_y)
residuals_Ad = np.array(residuals_Ad)

# Regression of residuals_y onto residuals_Ad with robust standard errors (HC3)
X_resid = sm.add_constant(residuals_Ad)
model = sm.OLS(residuals_y, X_resid).fit(cov_type='HC3')

# Display results
print(model.summary())

