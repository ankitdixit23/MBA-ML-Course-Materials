# Run using python 3.11
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import shap
import os
import requests
import rdata

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Switch to show or save files
dpl = False  # True = display plots, False = save plots

DROPBOX_URL = "https://www.dropbox.com/scl/fi/lcbyqfmkfxrhom7xxw5zb/RetailData.RData?rlkey=5cjn4u2od2c7yp4x16ccz4yfa&st=vemph0xk&dl=1"

resp = requests.get(DROPBOX_URL)
resp.raise_for_status()

parsed = rdata.parser.parse_data(resp.content)
converted = rdata.conversion.convert(parsed)

print(converted.keys())  
data = converted['Final.Data']

# Print names of all columns
for col in data.columns:
    print(f"{col}: {data[col].dtype}")

######################################################
# Descriptive statistics for the spending variable
print(data[['Spending']].describe())

##########################################################
# Let's look at average spending depending on whether the customer has a
# store credit card

# Calculate mean spending for each group
group_means = data.groupby("Cust.Has.XYZ.Company.CC")["Spending"].mean()

print(f"Mean with credit card: {group_means[1]}")
print(f"Mean without credit card: {group_means[0]}")
print(f"Difference: {group_means[1] - group_means[0]}")

# Calculate standard error of difference
group_stds = data.groupby("Cust.Has.XYZ.Company.CC")["Spending"].std()
group_sizes = data.groupby("Cust.Has.XYZ.Company.CC")["Spending"].count()

se_diff = np.sqrt(group_stds[1]**2/group_sizes[1] + group_stds[0]**2/group_sizes[0])

print(f"Standard error of difference: {se_diff}")


# Let's look at average spending depending on whether the customer was
# randomized to receive the promotion

# Calculate mean spending for each group
group_means = data.groupby("Promotion", observed = True)["Spending"].mean()

print(f"Mean with promotion: {group_means.iloc[1]}")
print(f"Mean without promotion: {group_means.iloc[0]}")
print(f"Difference: {group_means.iloc[1] - group_means.iloc[0]}")

# Calculate standard error of difference
group_stds = data.groupby("Promotion", observed = True)["Spending"].std()
group_sizes = data.groupby("Promotion", observed = True)["Spending"].count()

se_diff = np.sqrt(group_stds.iloc[1]**2/group_sizes.iloc[1] + group_stds.iloc[0]**2/group_sizes.iloc[0])

print(f"Standard error of difference: {se_diff}")


#########################################################################
# Heterogeneous effects and targeting
#########################################################################
# Drop Spending.Clusters because its a function of Spending
# Drop customer ID
#data.drop(columns=["Spending.Clusters", "Cust.ID"], inplace=True)
data.drop(columns=["Cust.ID"], inplace=True)

# Recode Promotion from "Treatment" and "Control" to 1/0
data['Promotion'] = data['Promotion'].map({'Control': 0, 'Treatment': 1}).astype(int)

# Sanity checking. I know the treatment fraction is 2/3
print(data['Promotion'].mean())

# Let's just set p to the known treatment probability
p = 2/3

# 1) Build models to predict Y
X = data.drop(columns=['Spending'])
y = data['Spending']

# Separate data into training and validation sets. We're only going to use the
# training data to choose our prediction models, choosing the model with the
# lowest cross-validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=219)

X_train0 = X_train[X_train['Promotion'] == 0]
y_train0 = y_train[X_train['Promotion'] == 0]
X_train1 = X_train[X_train['Promotion'] == 1]
y_train1 = y_train[X_train['Promotion'] == 1]
X_train0 = X_train0.drop(columns=['Promotion'])
X_train1 = X_train1.drop(columns=['Promotion'])

X_val0 = X_val[X_val['Promotion'] == 0]
y_val0 = y_val[X_val['Promotion'] == 0]
X_val1 = X_val[X_val['Promotion'] == 1]
y_val1 = y_val[X_val['Promotion'] == 1]
X_val0 = X_val0.drop(columns=['Promotion'])
X_val1 = X_val1.drop(columns=['Promotion'])

# Helper functions
# LASSO REGRESSION
def lasso_regression(X, y, title_suffix, dpl=dpl):
    # Create a pipeline with standardization and LassoCV
    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=5, random_state=42, n_jobs=-1))
    ])

    # Train the pipeline
    lasso_pipeline.fit(X, y)

    # Get the Lasso model from the pipeline
    lasso_model = lasso_pipeline.named_steps['lasso']

    # Get CV minimum MSE
    lasso_cvmse = np.min(lasso_model.mse_path_.mean(axis=1))

    # Plot Lasso Cross-Validation MSEs
    plt.figure()
    plt.plot(lasso_model.alphas_, lasso_model.mse_path_.mean(axis=1), marker='o')
    plt.xlabel('Alpha')
    plt.ylabel(f'Cross-Validated MSE, {title_suffix}')
    plt.title(f'Lasso Regression Cross-Validation MSE ({title_suffix})')
    plt.xscale('log')
    if dpl:
        plt.show()
    else:
        plt.savefig(os.path.join(script_dir, f"lasso_{title_suffix.replace('|', '_').replace(' ', '_')}.png"))
        plt.close()

    return lasso_pipeline, lasso_cvmse

# TREE REGRESSION
def tree_regression(X, y, title_suffix, dpl=dpl):

    # Parameters to search over
    dt_params = {'max_leaf_nodes': [2,3,4,5,10,15,20,25,30]}

    dt = GridSearchCV(DecisionTreeRegressor(random_state=42, min_samples_leaf=100),
                      dt_params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    dt.fit(X, y)

    # Get CV minimum MSE
    dt_cvmse = -dt.cv_results_['mean_test_score'][np.argmin(dt.cv_results_['mean_test_score'])]

    # Plot Decision Tree Cross-Validation MSE
    plt.figure()
    plt.plot(dt_params['max_leaf_nodes'], -dt.cv_results_['mean_test_score'], marker='o')
    plt.xlabel('Max Leaf Nodes')
    plt.ylabel(f'Cross-Validated MSE, {title_suffix}')
    plt.title(f'Decision Tree Cross-Validation MSE ({title_suffix})')
    if dpl:
        plt.show()
    else:
        plt.savefig(os.path.join(script_dir, f"tree_{title_suffix.replace('|', '_').replace(' ', '_')}.png"))
        plt.close()
    
    return dt, dt_cvmse

# RANDOM FOREST REGRESSION
def rf_regression(X, y, title_suffix, dpl=dpl):

    min_samples_leaf_values = [10, 25, 50, 100, 200]
    oob_errors = []
    best_rf = None
    best_oob_error = float('inf')

    # Train Random Forest with different min_samples_leaf values
    for min_samples_leaf in min_samples_leaf_values:
        rf = RandomForestRegressor(
            n_jobs=-1,
            n_estimators=100,
            random_state=42,
            oob_score=mean_squared_error,  # Use out-of-bag error
            min_samples_leaf=min_samples_leaf,
            max_features=0.33,  # Subsample 33% of the features per tree
            max_samples=0.5  # Subsample 50% of the data per tree
        )
        rf.fit(X, y)

        # Compute OOB error estimate
        oob_error = rf.oob_score_
        oob_errors.append(oob_error)

        if oob_error < best_oob_error:
            best_oob_error = oob_error
            best_rf = rf

        # Print progress
        print(f"min_samples_leaf: {min_samples_leaf}, OOB MSE: {oob_error:.4f}")

    # Plot OOB Error vs Min Samples per Leaf
    plt.figure()
    plt.plot(min_samples_leaf_values, oob_errors, marker='o', linestyle='-')
    plt.xlabel('Min Samples per Leaf')
    plt.ylabel(f'OOB MSE, {title_suffix}')
    plt.title(f'Random Forest OOB Error, {title_suffix}')
    plt.xticks(min_samples_leaf_values)
    plt.grid(True)
    if dpl:
        plt.show()
    else:
        plt.savefig(os.path.join(script_dir, f"rf_{title_suffix.replace('|', '_').replace(' ', '_')}.png"))
        plt.close()

    return best_rf, best_oob_error

# Estimate models
# ---- Lasso Regression with Cross-Validation ----
lasso0, lasso0_mse = lasso_regression(X_train0, y_train0, "Spending|Promotion = 0")
lasso1, lasso1_mse = lasso_regression(X_train1, y_train1, "Spending|Promotion = 1")

# ---- Tree Regression with Cross-Validation ----
tree0, tree0_mse = tree_regression(X_train0, y_train0, "Spending|Promotion = 0")
tree1, tree1_mse = tree_regression(X_train1, y_train1, "Spending|Promotion = 1")

# ---- Random Forest Regression ----
rf0, rf0_mse = rf_regression(X_train0, y_train0, "Spending|Promotion = 0")
rf1, rf1_mse = rf_regression(X_train1, y_train1, "Spending|Promotion = 1")

# Summarize
print(f"Lasso MSE: {lasso0_mse}, {lasso1_mse}")
print(f"Tree MSE: {tree0_mse}, {tree1_mse}")
print(f"RF MSE: {rf0_mse}, {rf1_mse}")


# 2) Estimate heterogeneous effects
# Just using lasso for simplicity
data['g0'] = lasso0.predict(X.drop(columns=['Promotion']))  # Obtain a prediction for each observation based on the T = 0 model
data['g1'] = lasso1.predict(X.drop(columns=['Promotion']))  # Obtain a prediction for each observation based on the T = 1 model

data['Ytilde'] = ((data['Promotion']/p - (1-data['Promotion'])/(1-p))*
                  (data['Spending'] - (1-data['Promotion'])*data['g0'] - data['Promotion']*data['g1'])
                  + data['g1'] - data['g0'])

X = data.drop(columns=['Spending', 'Promotion', 'Ytilde', 'g0', 'g1'])
y = data['Ytilde']

# Separate data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=219)

# Try different models for the CATE
lasso_cate, lasso_cate_mse = lasso_regression(X_train, y_train, "Ytilde")

# Out-of-sample prediction of \tilde{Y}
lasso_cate_pred = lasso_cate.predict(X_val)
lasso_cate_valmse = mean_squared_error(y_val, lasso_cate_pred)

tree_cate, tree_cate_mse = tree_regression(X_train, y_train, "Ytilde")

# Out-of-sample prediction of \tilde{Y}
tree_cate_pred = tree_cate.predict(X_val)
tree_cate_valmse = mean_squared_error(y_val, tree_cate_pred)

rf_cate, rf_cate_mse = rf_regression(X_train, y_train, "Ytilde")

# Out-of-sample prediction of \tilde{Y}
rf_cate_pred = rf_cate.predict(X_val)
rf_cate_valmse = mean_squared_error(y_val, rf_cate_pred)

# Summarize
print(f"Lasso CATE MSE: {lasso_cate_mse}, {lasso_cate_valmse}")
print(f"Tree CATE MSE: {tree_cate_mse}, {tree_cate_valmse}")
print(f"RF CATE MSE: {rf_cate_mse}, {rf_cate_valmse}")

# Interpret estimated CATE
# Lasso model
# Get the feature names
feature_names = X_train.columns

# Get the coefficients
lasso_coeffs = lasso_cate.named_steps['lasso'].coef_

# Create a DataFrame to display the results
lasso_results = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso_coeffs})

# Filter out features with zero coefficients (i.e., those not selected by Lasso)
selected_features = lasso_results[lasso_results['Coefficient'] != 0]

print(selected_features)

# SHAP
explainer = shap.Explainer(lasso_cate.predict, X_train, max_evals=1000)
small_X_val = X_val.sample(n=1000, random_state=42)
shap_values = explainer(small_X_val)

shap.plots.bar(shap_values, show=dpl)
if not dpl:
    plt.savefig(os.path.join(script_dir, "lasso_shap_importance.png"))
    plt.close()

shap_names = X_train.columns[np.argsort(np.abs(shap_values.values).mean(0))][::-1]
shap_top10 = shap_names[0:10]

shap.plots.scatter(shap_values[:, shap_top10[0:5].to_list()], show=dpl)
if not dpl:
    plt.savefig(os.path.join(script_dir, "lasso_shap_top5.png"))
    plt.close()
shap.plots.scatter(shap_values[:, shap_top10[5:10].to_list()], show=dpl)
if not dpl:
    plt.savefig(os.path.join(script_dir, "lasso_shap_next5.png"))
    plt.close()

# Random forest model
# rf importance built-in
rf_imp = pd.Series(rf_cate.feature_importances_, index=X_train.columns)
rf_imp.sort_values(ascending=True, inplace=True)
top_10_rf_imp = rf_imp.tail(10)
top_10_rf_imp.plot.barh(color='green')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance (Top 10)- Built-in Method")
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "rf_importance_builtin.png"))
    plt.close()

# SHAP
explainer = shap.Explainer(rf_cate)
small_X_val = X_val.sample(n=1000, random_state=42)
shap_values = explainer(small_X_val, check_additivity=False)

shap.plots.bar(shap_values, show=dpl)
if not dpl:
    plt.savefig(os.path.join(script_dir, "rf_shap_importance.png"))
    plt.close()

shap_names = X_train.columns[np.argsort(np.abs(shap_values.values).mean(0))][::-1]
shap_top10 = shap_names[0:10]

shap.plots.scatter(shap_values[:, shap_top10[0:5].to_list()], show=dpl)
if not dpl:
    plt.savefig(os.path.join(script_dir, "rf_shap_top5.png"))
    plt.close()
shap.plots.scatter(shap_values[:, shap_top10[5:10].to_list()], show=dpl)
if not dpl:
    plt.savefig(os.path.join(script_dir, "rf_shap_next5.png"))
    plt.close()

# 3) Evaluation based on value of targeting
# Form policy based on estimated CATE - assuming costs are 0
lasso_pol = lasso_cate_pred > 0
tree_pol = tree_cate_pred > 0
rf_pol = rf_cate_pred > 0

# Estimated expected value of targeting based on CATE
lasso_v = np.mean(lasso_pol*y_val)
tree_v = np.mean(tree_pol*y_val)
rf_v = np.mean(rf_pol*y_val)

# Display the estimates
print(f"Lasso V(0): {lasso_v}; Number treated: {np.sum(lasso_pol)}")
print(f"Tree V(0): {tree_v}; Number treated: {np.sum(tree_pol)}")
print(f"RF V(0): {rf_v}; Number treated: {np.sum(rf_pol)}")
print(f"Everyone V(0): {np.mean(y_val)}; Number treated: {len(y_val)}")

# For fun, let's suppose the cost is `treatment_cost`
treatment_cost = 2
lasso_pol = lasso_cate_pred > treatment_cost
tree_pol = tree_cate_pred > treatment_cost
rf_pol = rf_cate_pred > treatment_cost

# Estimated expected value of targeting based on CATE
lasso_v1 = np.mean(lasso_pol*(y_val-treatment_cost))
tree_v1 = np.mean(tree_pol*(y_val-treatment_cost))
rf_v1 = np.mean(rf_pol*(y_val-treatment_cost))

# Display the estimates
print(f"Lasso V(c): {lasso_v1}; Number treated: {np.sum(lasso_pol)}")
print(f"Tree V(c): {tree_v1}; Number treated: {np.sum(tree_pol)}")
print(f"RF V(c): {rf_v1}; Number treated: {np.sum(rf_pol)}")
print(f"Everyone V(0): {np.mean(y_val-treatment_cost)}; Number treated: {len(y_val)}")

#########################################################################
# Direct estimation of treatment function
# Assuming cost is 0
# Use rule we know is easy to interpret
policy0 = DecisionTreeClassifier(max_depth=2, min_impurity_decrease=1e-3,
                                min_samples_leaf=100,
                                random_state=42)
policy0.fit(X_train, y_train > 0, sample_weight=np.abs(y_train))

fig, ax = plt.subplots(1,1,figsize=(20,10))
plot_tree(policy0, filled=True, feature_names=list(X_train.columns), impurity=False, label='root',
          class_names=['Negative', 'Positive'], fontsize=10)
ax.set_title("Estimated Policy (Cost = 0)")
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "policy_cost0.png"))
    plt.close()

pi0 = (y_val) * policy0.predict(X_val)
point = np.mean(pi0)
stderr = np.sqrt(np.var(pi0) / pi0.shape[0])
print(f"Expected value targeting (cost = 0): {point:.5f}, {stderr:.5f}; number treated = {np.sum(policy0.predict(X_val))}")

pointA = np.mean(y_val)
stderrA = np.sqrt(np.var(y_val) / y_val.shape[0])
print(f"Expected value everyone (cost = 0): {pointA:.5f}, {stderrA:.5f}")

# Assuming cost is `treatment_cost`
policyC = DecisionTreeClassifier(max_depth=2, min_impurity_decrease=1e-3,
                                min_samples_leaf=100,
                                random_state=42)
policyC.fit(X_train, (y_train - treatment_cost > 0), sample_weight=np.abs(y_train - treatment_cost))

fig, ax = plt.subplots(1,1,figsize=(20,10))
plot_tree(policyC, filled=True, feature_names=list(X_train.columns), impurity=False, label='root',
          class_names=['Negative', 'Positive'], fontsize=10)
ax.set_title(f"Estimated Policy (Cost = {treatment_cost})")
if dpl: 
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, f"policy_cost{treatment_cost}.png"))
    plt.close()

piC = (y_val - treatment_cost) * policyC.predict(X_val)
pointC = np.mean(piC)
stderrC = np.sqrt(np.var(piC) / piC.shape[0])
print(f"Expected value targeting (cost = {treatment_cost}): {pointC:.5f}, {stderrC:.5f}; number treated = {np.sum(policyC.predict(X_val))}")

pointAC = np.mean(y_val - treatment_cost)
stderrAC = np.sqrt(np.var(y_val - treatment_cost) / y_val.shape[0])
print(f"Expected value everyone (cost = {treatment_cost}): {pointAC:.5f}, {stderrAC:.5f}")
