# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
# Note: h2o requires a java runtime environment to be installed
import h2o
from h2o.automl import H2OAutoML
import requests
import os
from io import BytesIO

# switch to turn off plots (Switch to True to turn on plots)
dpl = False

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Download the house price data
dropbox_url = "https://www.dropbox.com/scl/fi/omkehbrmstg78xv1tynga/bank-additional-full.csv?rlkey=c5uuvoujyg9vzdiysv7ugin61&dl=1"
response = requests.get(dropbox_url)
rawdata = pd.read_csv(BytesIO(response.content), sep = ";")
print(rawdata.head())

pd.set_option('future.no_silent_downcasting', True)

# Recode outcome from "yes" and "no" to 1 and 0
rawdata['y'] = rawdata['y'].replace({'no': 0, 'yes': 1})

# Duration is essentially an outcome. It is unknown before the contact and y = 0 => duration = 0
# Drop duration column
rawdata = rawdata.drop(columns = ['duration'])

# Create variable indicating not previously contacted and replace 999's in pdays with 0's
rawdata['never_contacted'] = np.where(rawdata['pdays'] == 999, 1, 0)
rawdata['never_contacted'] = rawdata['never_contacted'].astype('category')
rawdata['pdays'] = np.where(rawdata['pdays'] == 999, 0, rawdata['pdays'])

# Split the data into training (80%) and validation (20%) sets
train, val = train_test_split(rawdata, test_size = 0.2, random_state = 94)

# For use with cross-validation later
cvsplit = KFold(n_splits = 5, shuffle = True, random_state = 94)

# Set data up for use with logistic regression
X_train = train.drop(columns = ['y'])
y_train = train['y']
X_val = val.drop(columns = ['y'])
y_val = val['y']

# Make sure y_train and y_val are binary type
y_train = y_train.astype(int)
y_val = y_val.astype(int)

# Constant model - ignore features
y_pred_constant = np.zeros(len(y_val)).astype(int)
y_pred_prob_constant = np.zeros(len(y_val))+np.mean(y_train)

# Evaluate the model
constant_classification_metrics = classification_report(y_val, y_pred_constant, output_dict=True)
print(pd.DataFrame(constant_classification_metrics))

# Cross-entropy
constant_cross_entropy = -np.mean(y_val * np.log(y_pred_prob_constant) + (1 - y_val) * np.log(1 - y_pred_prob_constant))
print(f'Cross-entropy (Constant) = {constant_cross_entropy:.4f}')

if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_constant)
    plt.show()

# Get dummy variables for our categorical features. Because we are not doing regularization or any kind of variable selection,
# we will drop one of each set of dummies.
X_train = pd.get_dummies(X_train, drop_first = True)
X_val = pd.get_dummies(X_val, drop_first = True)

# Align validation data to ensure same columns as training data
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

# Fitting the logistic regression
logistic_model = LogisticRegression(max_iter=1000, random_state=94, penalty = None)
logistic_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_logistic = logistic_model.predict(X_val)
y_pred_prob_logistic = logistic_model.predict_proba(X_val)[:, 1]

# Evaluate the model
logistic_classification_metrics = classification_report(y_val, y_pred_logistic, output_dict=True)
print(pd.DataFrame(logistic_classification_metrics))

# Cross entropy
logistic_cross_entropy = -np.mean(y_val * np.log(y_pred_prob_logistic) + (1 - y_val) * np.log(1 - y_pred_prob_logistic))
print(f'Cross-entropy (Logistic) = {logistic_cross_entropy:.4f}')

if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_logistic)
    plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob_logistic)
roc_auc_logistic = roc_auc_score(y_val, y_pred_prob_logistic)

fpr_con, tpr_con, thresholds_con = roc_curve(y_val, y_pred_prob_constant)
roc_auc_constant = roc_auc_score(y_val, y_pred_prob_constant)

if dpl:
    plt.plot(fpr, tpr, label=f'Logistic Regression (area = {roc_auc_logistic:.2f})')
    plt.plot(fpr_con, tpr_con, label=f'Constant (area = {roc_auc_constant:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Cumulative Gain
# Creating a DataFrame with the true values and predicted probabilities
data = pd.DataFrame({'true': y_val, 'prob': y_pred_prob_logistic})
data.sort_values(by='prob', ascending=False, inplace=True)

# Calculating cumulative gain
data['cumulative_gain'] = np.cumsum(data['true']) / data['true'].sum()
data['cumulative_percentage'] = np.arange(1, len(data) + 1) / len(data)

# Plotting
if dpl:
    plt.figure(figsize=(10, 6))
    plt.plot(data['cumulative_percentage'], data['cumulative_gain'], label='Cumulative Gain')
    plt.plot([0, 1], [0, 1], 'r--', label='Baseline')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Cumulative gain')
    plt.title('Cumulative Gain Chart (Logistic)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Lift
# Calculating lift
data['lift'] = data['cumulative_gain'] / data['cumulative_percentage']

# Plotting lift curve
if dpl:
    plt.figure(figsize=(10, 6))
    plt.plot(data['cumulative_percentage'], data['lift'], label='Lift Curve')
    plt.plot([0, 1], [1, 1], 'r--')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Lift')
    plt.title('Lift Curve (Logistic)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Let's look at classification trees
# Start with simplest possible
tree1 = DecisionTreeClassifier(max_leaf_nodes = 2)
tree1.fit(X_train, y_train)
if dpl:
    plot_tree(tree1, feature_names=X_train.columns)
    plt.show()

# Now let's look at 2 splits (3 leaves)
tree2 = DecisionTreeClassifier(max_leaf_nodes = 3)
tree2.fit(X_train, y_train)
if dpl:
    plot_tree(tree2, feature_names=X_train.columns)
    plt.show()

# Make the figure bigger than default so it's easier to read
if dpl:
    width = 10
    height = 7
    plt.figure(figsize=(width, height))

# Let's fit a tree with 3 splits = 4 leaves
tree3 = DecisionTreeClassifier(max_leaf_nodes = 4)
tree3.fit(X_train, y_train)
if dpl:
    plot_tree(tree3, feature_names=X_train.columns)
    plt.show()

# Now let's choose the number of leaves via cross-validation using the training data

# Parameter we want to choose based on cross-validation performance - number of leaves
parameters = {'max_leaf_nodes':range(2,51)}

# Define model and do cross-validation
tree = DecisionTreeClassifier()
cv_tree = GridSearchCV(tree, parameters,
                       n_jobs=-1, scoring='neg_log_loss', refit=True, cv=cvsplit)
# We can evaluate our performance based on many different measures. We're using
# cross-entropy (aka log loss) in this example.
# The commented lines below uses accuracy or recall instead.
#cv_tree = GridSearchCV(tree, parameters, n_jobs=-1, scoring='accuracy', refit=True, cv=cvsplit)
#cv_tree = GridSearchCV(tree, parameters, n_jobs=-1, scoring='recall', refit=True, cv=cvsplit)

# Perform cross validation
cv_tree.fit(X_train, y_train)

# Make the figure bigger than default so it's easier to read
if dpl:
    width = 20
    height = 20
    plt.figure(figsize=(width, height))

# Pull out and plot the tree corresponding to the best prediction rule
# according to CV.
best_tree = cv_tree.best_estimator_

if dpl:
    plot_tree(best_tree, feature_names = X_train.columns)
    plt.show()

leaves = cv_tree.cv_results_.get('param_max_leaf_nodes')
leaves = leaves.tolist()

lranks = cv_tree.cv_results_.get('rank_test_score')
loss = -cv_tree.cv_results_.get('mean_test_score')

if dpl:
    plt.plot(leaves, loss, label = 'Loss')
    plt.axvline(cv_tree.best_params_.get('max_leaf_nodes'),
                linestyle="--", color="black", label="CV estimate")
    plt.xlabel("Number of leaves")
    plt.ylabel("Cross-validation Performance")
    plt.legend()
    plt.show()

# Performance on the validation data
y_pred_tree = best_tree.predict(X_val)
y_pred_prob_tree = best_tree.predict_proba(X_val)[:, 1]

# Evaluate the model
tree_classification_metrics = classification_report(y_val, y_pred_tree, output_dict=True)
print(pd.DataFrame(tree_classification_metrics))

# Cross entropy
tree_cross_entropy = -np.mean(y_val * np.log(y_pred_prob_tree) + (1 - y_val) * np.log(1 - y_pred_prob_tree))
print(f'Cross-entropy (Tree) = {tree_cross_entropy:.4f}')

# Display confusion matrix
if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_tree)
    plt.show()

# Random forest
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 250, 500, 1000],
    'min_samples_leaf': [1, 15, 30, 60, 120]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=94)

# Cross validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cvsplit, scoring='neg_log_loss', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Create dataframe with cv metrics
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['mean_test_score'] = -cv_results['mean_test_score']
print(cv_results[['param_min_samples_leaf', 'param_n_estimators', 'mean_test_score']])

# Get best model (based on cv)
best_rf = grid_search.best_estimator_

# Performance on the validation data
y_pred_rf = best_rf.predict(X_val)
y_pred_prob_rf = best_rf.predict_proba(X_val)[:, 1]

# Evaluate the model
rf_classification_metrics = classification_report(y_val, y_pred_rf, output_dict=True)
print(pd.DataFrame(rf_classification_metrics))

# Cross-entropy
rf_cross_entropy = -np.mean(y_val * np.log(y_pred_prob_rf) + (1 - y_val) * np.log(1 - y_pred_prob_rf))
print(f'Cross-entropy (RF) = {rf_cross_entropy:.4f}')

# Display confusion matrix
if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_rf)
    plt.show()

# Let's try with boosting
# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [2, 3, 4, 5, 6],
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

# Initialize the GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=94)

grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=cvsplit, scoring='neg_log_loss', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Create dataframe with cv metrics
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['mean_test_score'] = -cv_results['mean_test_score']

# Plot cv results with separate lines for each learning rate/depth combination
if dpl:
    plt.figure()
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            mask = (cv_results['param_learning_rate'] == learning_rate) & (cv_results['param_max_depth'] == max_depth)
            plt.plot(cv_results[mask]['param_n_estimators'], cv_results[mask]['mean_test_score'], label=f'lr={learning_rate}, md={max_depth}')

    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('mean_test_score')
    plt.show()

# Get best model based on cv
best_gbc = grid_search.best_estimator_

# Performance on validation data
y_pred_gbc = best_gbc.predict(X_val)
y_pred_prob_gbc = best_gbc.predict_proba(X_val)[:, 1]

# Cross entropy
gbc_cross_entropy = -np.mean(y_val * np.log(y_pred_prob_gbc) + (1 - y_val) * np.log(1 - y_pred_prob_gbc))
print(f'Cross-entropy (GBC) = {gbc_cross_entropy:.4f}')

# Evaluate the model
gbc_classification_metrics = classification_report(y_val, y_pred_gbc, output_dict=True)
print(pd.DataFrame(gbc_classification_metrics))

# Display confusion matrix
if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_gbc)
    plt.show()

# Redo GBC using early stopping
# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [2, 3, 4, 5, 6]
}

# Early stopping criteria
best_model = None
best_score = float('inf')

for learning_rate in param_grid['learning_rate']:
    for max_depth in param_grid['max_depth']:
        # Initialize the GradientBoostingClassifier with early stopping
        gbcES = GradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=200,
            validation_fraction=0.2,
            n_iter_no_change=10,
            tol=1e-4,
            random_state=94
        )

        # Fit the model
        gbcES.fit(X_train, y_train)

        # Evaluate the validation loss from the internal validation split
        val_loss = min(gbcES.train_score_[-10:])  # Use the lowest validation loss from early stopping

        if val_loss < best_score:
            best_score = val_loss
            best_model = gbcES

# Print the best parameters
print(f"Best model parameters: learning_rate={best_model.learning_rate}, max_depth={best_model.max_depth}")

# Evaluate the best model on the test data
y_pred_gbcES = best_model.predict(X_val)
y_pred_prob_gbcES = best_model.predict_proba(X_val)[:, 1]

# Evaluate the model
gbcES_classification_metrics = classification_report(y_val, y_pred_gbcES, output_dict=True)
print(pd.DataFrame(gbcES_classification_metrics))

# Cross entropy
gbcES_cross_entropy = -np.mean(y_val * np.log(y_pred_prob_gbcES) + (1 - y_val) * np.log(1 - y_pred_prob_gbcES))
print(f'Cross-entropy (GBCES) = {gbcES_cross_entropy:.4f}')

if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_gbcES)
    plt.show()

# Retrain using the best parameters without early stopping
long_model = GradientBoostingClassifier(
    learning_rate=best_gbc.learning_rate,
    max_depth=best_gbc.max_depth,
    n_estimators=200,
    random_state=94
)
long_model.fit(X_train, y_train)


train_acc_without = []
val_acc_without = []

train_acc_with = []
val_acc_with = []

for i, (train_pred, val_pred) in enumerate(
    zip(
        long_model.staged_predict(X_train),
        long_model.staged_predict(X_val),
    )
):
    train_acc_without.append(accuracy_score(y_train, train_pred))
    val_acc_without.append(accuracy_score(y_val, val_pred))

for i, (train_pred, val_pred) in enumerate(
    zip(
        best_model.staged_predict(X_train),
        best_model.staged_predict(X_val),
    )
):
    train_acc_with.append(accuracy_score(y_train, train_pred))
    val_acc_with.append(accuracy_score(y_val, val_pred))



if dpl:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

    axes[0].plot(train_acc_without, label="gbm_full")
    axes[0].plot(train_acc_with, label="gbm_early_stopping")
    axes[0].set_xlabel("Boosting Iterations")
    axes[0].set_ylabel("Accuracy (Training)")
    axes[0].legend()
    axes[0].set_title("Training Accuracy")

    axes[1].plot(val_acc_without, label="gbm_full")
    axes[1].plot(val_acc_with, label="gbm_early_stopping")
    axes[1].set_xlabel("Boosting Iterations")
    axes[1].set_ylabel("Accuracy (Validation)")
    axes[1].legend()
    axes[1].set_title("Validation Accuracy")

    plt.show()

# Evaluate on the basis of an "economic" metric
#For simplicity, let's suppose that there is a cost  C=100  of contacting an individual.
#
#Let's suppose that a customer who is contacted and opens an account has the following characteristics:
#
#Initial deposit = 1000, average maintained balance = 2000
#Assessed fees = 50/year
#Takes out credit card that generates 100 in fees and interest/year
#Stays with the bank for 5 years
#
#Let's further assume that the bank
#
#gets 5%/year in interest from loans made from deposits
#incurs costs of 100 initially and 20/year for account maintenance, setup, ...
#has a discount rate of 5%
#
#From this, we have that the net revenue of the customer is (100+50+100-100-20 = 130) in year 1
#and (100+50+100-20 = 230) in years 2-5.
#Lifetime value of the customer is then approximately  R=(130)/(1.05)+sum(j=2,..,5)(230)(1.05^j)=904~900.
#
# Expected benefit - assuming we get TP and FP exactly as in our predictions -
# P(TP)*(R-C)+P(FP)*(-C)

# Assume "contacting" anyone with predicted take-up > .5
# Confusion matrix for constant model
tn, fp, fn, tp = confusion_matrix(y_val, y_pred_constant).ravel()
# Confusion matrix for logistic model
tn_log, fp_log, fn_log, tp_log = confusion_matrix(y_val, y_pred_logistic).ravel()
# Confusion matrix for tree model
tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_val, y_pred_tree).ravel()
# Confusion matrix for random forest model
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_val, y_pred_rf).ravel()
# Confusion matrix for boosting model
tn_gbc, fp_gbc, fn_gbc, tp_gbc = confusion_matrix(y_val, y_pred_gbc).ravel()
# Confusion matrix for boosting model (early stopping)
tn_gbcES, fp_gbcES, fn_gbcES, tp_gbcES = confusion_matrix(y_val, y_pred_gbcES).ravel()

# Hypothetical value of customer opening deposit
R = 900
# Hypothetical "benefit" of contacting customer
C = -100

# Expected profit from constant model
N = tn+fp+fn+tp  # Total number of observations
Epi_con = (tp/N)*(R+C)+(fp/N)*C
# Expected profit from logistic model
Epi_log = (tp_log/N)*(R+C)+(fp_log/N)*C
# Expected profit from tree model
Epi_tr = (tp_tr/N)*(R+C)+(fp_tr/N)*C
# Expected profit from random forest model
Epi_rf = (tp_rf/N)*(R+C)+(fp_rf/N)*C
# Expected profit from boosting model
Epi_gbc = (tp_gbc/N)*(R+C)+(fp_gbc/N)*C
# Expected profit from boosting model (early stopping)
Epi_gbcES = (tp_gbcES/N)*(R+C)+(fp_gbcES/N)*C


print(f"Expected profit from constant model: {Epi_con}")
print(f"Expected profit from logistic model: {Epi_log}")
print(f"Expected profit from tree model: {Epi_tr}")
print(f"Expected profit from random forest model: {Epi_rf}")
print(f"Expected profit from boosted tree model: {Epi_gbc}")
print(f"Expected profit from boosted tree model (early stopping): {Epi_gbcES}")

# Consider choosing threshold for contact based on
# 1. First, order all individuals in the sample according to predicted probability of take-up.
# 2. Consider varying the threshold for contact (or alternatively the fraction of the sample to contact) between 1 (contact no one) and 0 (contact everyone)
# 3. Choose the model and threshold that gives the highest "profit." (All the true ones give a benefit of R-C, and all the true zeros give a benefit of -C.)
# Creating a DataFrame with the true values and predicted probabilities
datalog = pd.DataFrame({'true': y_val, 'prob': y_pred_prob_logistic})
datalog.sort_values(by='prob', ascending=False, inplace=True)

datatr = pd.DataFrame({'true': y_val, 'prob': y_pred_prob_tree})
datatr.sort_values(by='prob', ascending=False, inplace=True)

datarf = pd.DataFrame({'true': y_val, 'prob': y_pred_prob_rf})
datarf.sort_values(by='prob', ascending=False, inplace=True)

datagbc = pd.DataFrame({'true': y_val, 'prob': y_pred_prob_gbc})
datagbc.sort_values(by='prob', ascending=False, inplace=True)

datagbcES = pd.DataFrame({'true': y_val, 'prob': y_pred_prob_gbcES})
datagbcES.sort_values(by='prob', ascending=False, inplace=True)

# Calculate "profit" from contacting people ordered by their predicted probability
datalog['profit'] = np.cumsum(datalog['true']*(R-C)+(1-datalog['true'])*C)
datalog['cumulative_percentage'] = np.arange(1, len(datalog) + 1) / len(datalog)

datatr['profit'] = np.cumsum(datatr['true']*(R-C)+(1-datatr['true'])*C)
datatr['cumulative_percentage'] = np.arange(1, len(datatr) + 1) / len(datatr)

datarf['profit'] = np.cumsum(datarf['true']*(R-C)+(1-datarf['true'])*C)
datarf['cumulative_percentage'] = np.arange(1, len(datarf) + 1) / len(datarf)

datagbc['profit'] = np.cumsum(datagbc['true']*(R-C)+(1-datagbc['true'])*C)
datagbc['cumulative_percentage'] = np.arange(1, len(datagbc) + 1) / len(datagbc)

datagbcES['profit'] = np.cumsum(datagbcES['true']*(R-C)+(1-datagbcES['true'])*C)
datagbcES['cumulative_percentage'] = np.arange(1, len(datagbcES) + 1) / len(datagbcES)

# Plotting
if dpl:
    plt.figure(figsize=(10, 6))
    plt.plot(datalog['cumulative_percentage'], datalog['profit'], label='Profit - Logistic')
    plt.plot(datarf['cumulative_percentage'], datatr['profit'],
            label='Profit - Tree')
    plt.plot(datarf['cumulative_percentage'], datarf['profit'],
            label='Profit - Random Forest')
    plt.plot(datagbc['cumulative_percentage'], datagbc['profit'],
            label='Profit - Boosted Tree')
    plt.plot(datagbcES['cumulative_percentage'], datagbcES['profit'],
            label='Profit - Boosted Tree (early stopping)')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Profit')
    plt.title('Profit Chart')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Get predicted probability in row corresponding to maximum profit (logistic)
logprob = datalog.loc[datalog['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (logistic): {logprob}")
logprof = datalog.loc[datalog['profit'].idxmax(),'profit']
print(f"Maximum profit (logistic): {logprof}")

# Let's calculate accuracy using this decision threshold
logpredictions = (y_pred_prob_logistic >= logprob).astype(int)
logaccuracy = np.mean(logpredictions == y_val)
print(f"Accuracy (logistic): {logaccuracy}")

# Get predicted probability in row corresponding to maximum profit (tree)
treeprob = datatr.loc[datatr['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (tree): {treeprob}")
treeprof = datatr.loc[datalog['profit'].idxmax(),'profit']
print(f"Maximum profit (tree): {treeprof}")

# Let's calculate accuracy using this decision threshold
treepredictions = (y_pred_prob_tree >= treeprob).astype(int)
treeaccuracy = np.mean(treepredictions == y_val)
print(f"Accuracy (tree): {treeaccuracy}")

# Get predicted probability in row corresponding to maximum profit (random forest)
rfprob = datarf.loc[datarf['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (random forest): {rfprob}")
rfprof = datarf.loc[datarf['profit'].idxmax(),'profit']
print(f"Maximum profit (random forest): {rfprof}")

# Let's calculate accuracy using this decision threshold
rfpredictions = (y_pred_prob_rf >= rfprob).astype(int)
rfaccuracy = np.mean(rfpredictions == y_val)
print(f"Accuracy (random forest): {rfaccuracy}")

# Get predicted probability in row corresponding to maximum profit (boosted trees)
gbcprob = datarf.loc[datagbc['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (boosted trees): {gbcprob}")
gbcprof = datarf.loc[datagbc['profit'].idxmax(),'profit']
print(f"Maximum profit (boosted trees): {gbcprof}")

# Let's calculate accuracy using this decision threshold
gbcpredictions = (y_pred_prob_gbc >= gbcprob).astype(int)
gbcaccuracy = np.mean(gbcpredictions == y_val)
print(f"Accuracy (boosted trees): {gbcaccuracy}")

# Get predicted probability in row corresponding to maximum profit (boosted trees, early)
gbcESprob = datarf.loc[datagbcES['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (boosted trees, early): {gbcESprob}")
gbcESprof = datarf.loc[datagbcES['profit'].idxmax(),'profit']
print(f"Maximum profit (boosted trees, early): {gbcESprof}")

# Let's calculate accuracy using this decision threshold
gbcESpredictions = (y_pred_prob_gbcES >= gbcESprob).astype(int)
gbcESaccuracy = np.mean(gbcESpredictions == y_val)
print(f"Accuracy (boosted trees, early): {gbcESaccuracy}")

# Stacking
# We are going to drop early stopping and stack based on cross-validated
# predicted values within the training data. We would need to do early
# stopping within cross-validation if we wanted to use it here.

cv_log = cross_val_predict(logistic_model, X_train, y_train,
                           cv=cvsplit, method='predict_proba', n_jobs = -1)
cv_tree = cross_val_predict(best_tree, X_train, y_train,
                            cv=cvsplit, method='predict_proba', n_jobs = -1)
cv_rf = cross_val_predict(best_rf, X_train, y_train,
                          cv=cvsplit, method='predict_proba', n_jobs = -1)
cv_gbc = cross_val_predict(best_gbc, X_train, y_train,
                           cv=cvsplit, method='predict_proba', n_jobs = -1)

# Linear regression of y_train on the cross-validated predictions
stack_model = LinearRegression()
stack_model.fit(np.column_stack((cv_log[:, 1], cv_tree[:, 1], cv_rf[:, 1], cv_gbc[:, 1])), y_train)

# Coefficients from fitted linear model
print(stack_model.coef_)
print(stack_model.intercept_)

# Use fitted model to obtain predicted values in validation data
y_pred_stack_prob = stack_model.predict(np.column_stack((y_pred_prob_logistic, y_pred_prob_tree, y_pred_prob_rf, y_pred_prob_gbc)))
y_pred_stack = (y_pred_stack_prob >= 0.5).astype(int)

# Evaluate stacking model
stack_classification_metrics = classification_report(y_val, y_pred_stack, output_dict=True)
print(pd.DataFrame(stack_classification_metrics))

# cross entropy
stack_cross_entropy = -np.mean(y_val * np.log(y_pred_stack_prob) + (1 - y_val) * np.log(1 - y_pred_stack_prob))
print(f'Cross-entropy (stack) = {stack_cross_entropy:.4f}')

# Confusion matrix
if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_stack)
    plt.show()

# Profit
# Confusion matrix for stacked model
tn_stack, fp_stack, fn_stack, tp_stack = confusion_matrix(y_val, y_pred_stack).ravel()

# Expected profit from stacked model
Epi_stack = (tp_stack/N)*(R+C)+(fp_stack/N)*C
print(f"Expected profit from stacked model: {Epi_stack}")

# Add stacked model to profit figure
# Creating a DataFrame with the true values and predicted probabilities
datastack = pd.DataFrame({'true': y_val, 'prob': y_pred_stack_prob})
datastack.sort_values(by='prob', ascending=False, inplace=True)

# Calculate "profit" from contacting people ordered by their predicted probability
datastack['profit'] = np.cumsum(datastack['true']*(R-C)+(1-datastack['true'])*C)
datastack['cumulative_percentage'] = np.arange(1, len(datastack) + 1) / len(datastack)

if dpl:
    # Figure
    plt.figure(figsize=(10, 6))
    plt.plot(datalog['cumulative_percentage'], datalog['profit'], label='Profit - Logistic')
    plt.plot(datarf['cumulative_percentage'], datatr['profit'],
             label='Profit - Tree')
    plt.plot(datarf['cumulative_percentage'], datarf['profit'],
             label='Profit - Random Forest')
    plt.plot(datagbc['cumulative_percentage'], datagbc['profit'],
             label='Profit - Boosted Tree')
    plt.plot(datagbcES['cumulative_percentage'], datagbcES['profit'],
             label='Profit - Boosted Tree (early stopping)')
    plt.plot(datastack['cumulative_percentage'], datastack['profit'],
             label='Profit - Stacking')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Profit')
    plt.title('Profit Chart')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Get predicted probability in row corresponding to maximum profit (stacking)
stackprob = datastack.loc[datastack['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (stacking): {stackprob}")
stackprof = datastack.loc[datastack['profit'].idxmax(),'profit']
print(f"Maximum profit (stacking): {stackprof}")

# Let's calculate accuracy using this decision threshold
stackpredictions = (y_pred_stack_prob >= stackprob).astype(int)
stackaccuracy = np.mean(stackpredictions == y_val)
print(f"Accuracy (stacking): {stackaccuracy}")

# For comparison, we are going to look at stacking using the validation data
# predicted values and all learners. We would ideally have a further testing
# sample set aside if we were using this approach though.

# Linear regression of y_val on the validation data predictions
val_stack_model = LinearRegression()
val_stack_model.fit(np.column_stack((cv_log[:, 1], cv_tree[:, 1], cv_rf[:, 1], cv_gbc[:, 1])), y_train)

# Coefficients from fitted linear model
print(val_stack_model.coef_)
print(val_stack_model.intercept_)

# Use fitted model to obtain predicted values in validation data
y_pred_val_stack_prob = val_stack_model.predict(np.column_stack((y_pred_prob_logistic, y_pred_prob_tree, y_pred_prob_rf, y_pred_prob_gbc)))
y_pred_val_stack = (y_pred_val_stack_prob >= 0.5).astype(int)

# Evaluate stacking model
val_stack_classification_metrics = classification_report(y_val, y_pred_val_stack, output_dict=True)
print(pd.DataFrame(val_stack_classification_metrics))

# cross entropy
val_stack_cross_entropy = -np.mean(y_val * np.log(y_pred_val_stack_prob) + (1 - y_val) * np.log(1 - y_pred_val_stack_prob))
print(f'Cross-entropy (val_stack) = {val_stack_cross_entropy:.4f}')

# Confusion matrix
if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val_stack)
    plt.show()

# Profit
# Confusion matrix for stacked model
tn_val_stack, fp_val_stack, fn_val_stack, tp_val_stack = confusion_matrix(y_val, y_pred_val_stack).ravel()

# Expected profit from stacked model
Epi_val_stack = (tp_val_stack/N)*(R+C)+(fp_val_stack/N)*C
print(f"Expected profit from val_stacked model: {Epi_val_stack}")

# Add stacked model to profit figure
# Creating a DataFrame with the true values and predicted probabilities
dataval_stack = pd.DataFrame({'true': y_val, 'prob': y_pred_val_stack_prob})
dataval_stack.sort_values(by='prob', ascending=False, inplace=True)

# Calculate "profit" from contacting people ordered by their predicted probability
dataval_stack['profit'] = np.cumsum(dataval_stack['true']*(R-C)+(1-dataval_stack['true'])*C)
dataval_stack['cumulative_percentage'] = np.arange(1, len(dataval_stack) + 1) / len(dataval_stack)

if dpl:
    # Figure
    plt.figure(figsize=(10, 6))
    plt.plot(datalog['cumulative_percentage'], datalog['profit'], label='Profit - Logistic')
    plt.plot(datarf['cumulative_percentage'], datatr['profit'],
             label='Profit - Tree')
    plt.plot(datarf['cumulative_percentage'], datarf['profit'],
             label='Profit - Random Forest')
    plt.plot(datagbc['cumulative_percentage'], datagbc['profit'],
             label='Profit - Boosted Tree')
    plt.plot(datagbcES['cumulative_percentage'], datagbcES['profit'],
             label='Profit - Boosted Tree (early stopping)')
    plt.plot(datastack['cumulative_percentage'], datastack['profit'],
             label='Profit - Stacking')
    plt.plot(dataval_stack['cumulative_percentage'], dataval_stack['profit'],
             label='Profit - (Validation data) Stacking')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Profit')
    plt.title('Profit Chart')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Get predicted probability in row corresponding to maximum profit (stacking)
val_stackprob = dataval_stack.loc[dataval_stack['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (val_stacking): {val_stackprob}")
val_stackprof = dataval_stack.loc[dataval_stack['profit'].idxmax(),'profit']
print(f"Maximum profit (val_stacking): {val_stackprof}")

# Let's calculate accuracy using this decision threshold
val_stackpredictions = (y_pred_val_stack_prob >= val_stackprob).astype(int)
val_stackaccuracy = np.mean(val_stackpredictions == y_val)
print(f"Accuracy (val_stacking): {val_stackaccuracy}")

h2o.init()

# Convert pandas DataFrame to H2OFrame
train_h2o = h2o.H2OFrame(train)
val_h2o = h2o.H2OFrame(val)

# Need to convert y variable to factor if going to use h2o
train_h2o['y'] = train_h2o['y'].asfactor()
val_h2o['y'] = val_h2o['y'].asfactor()

# Data
X_train = train.drop(columns=['y'])
y_train = train['y']
X_val = val.drop(columns=['y'])
y_val = val['y']

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=42, nfolds = 5)
aml.train(x=list(X_train.columns), y=y_train.name, training_frame=train_h2o)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)

# Print name of best model
print(aml.leader)

# Evaluate the best model on the test data
y_pred_aml_all = aml.leader.predict(val_h2o)  # Returns 3 columns.
# First column = binary prediction
# Second column = probability y = 0
# Third column = probability y = 1

y_pred_aml = y_pred_aml_all[:,0].as_data_frame(use_pandas=True).to_numpy().ravel().astype(int)
y_pred_prob_aml = y_pred_aml_all[:,2].as_data_frame(use_pandas=True).to_numpy().ravel()

# Cross-entropy
aml_cross_entropy = -np.mean(y_val * np.log(y_pred_prob_aml) + (1 - y_val) * np.log(1 - y_pred_prob_aml))
print(f'Cross-entropy (h2o) = {aml_cross_entropy:.4f}')

# Convert y_val to integer type
y_val = y_val.astype(int)

# Evaluate the model
aml_classification_metrics = classification_report(y_val, y_pred_aml, output_dict=True)
print(pd.DataFrame(aml_classification_metrics))

if dpl:
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_aml)
    plt.show()

# Profit
# Confusion matrix for h2o model
tn_aml, fp_aml, fn_aml, tp_aml = confusion_matrix(y_val, y_pred_aml).ravel()

# Expected profit from h2o model
Epi_aml = (tp_aml/N)*(R+C)+(fp_aml/N)*C
print(f"Expected profit from h2o model: {Epi_aml}")

# Add h2o model to profit figure
# Creating a DataFrame with the true values and predicted probabilities
dataaml = pd.DataFrame({'true': y_val, 'prob': y_pred_prob_aml})
dataaml.sort_values(by='prob', ascending=False, inplace=True)

# Calculate "profit" from contacting people ordered by their predicted probability
dataaml['profit'] = np.cumsum(dataaml['true']*(R-C)+(1-dataaml['true'])*C)
dataaml['cumulative_percentage'] = np.arange(1, len(dataaml) + 1) / len(dataaml)

if dpl:
    # Figure
    plt.figure(figsize=(10, 6))
    plt.plot(datalog['cumulative_percentage'], datalog['profit'], label='Profit - Logistic')
    plt.plot(datarf['cumulative_percentage'], datatr['profit'],
             label='Profit - Tree')
    plt.plot(datarf['cumulative_percentage'], datarf['profit'],
             label='Profit - Random Forest')
    plt.plot(datagbc['cumulative_percentage'], datagbc['profit'],
             label='Profit - Boosted Tree')
    plt.plot(datagbcES['cumulative_percentage'], datagbcES['profit'],
             label='Profit - Boosted Tree (early stopping)')
    plt.plot(datastack['cumulative_percentage'], datastack['profit'],
             label='Profit - Stacking')
    plt.plot(dataval_stack['cumulative_percentage'], dataval_stack['profit'],
             label='Profit - (Validation data) Stacking')
    plt.plot(dataaml['cumulative_percentage'], dataaml['profit'],
             label='Profit - h2o')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Profit')
    plt.title('Profit Chart')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Get predicted probability in row corresponding to maximum profit (h2o)
amlprob = dataaml.loc[dataaml['profit'].idxmax(),'prob']
print(f"Predicted takeup probability threshold (h2o): {amlprob}")
amlprof = dataaml.loc[dataaml['profit'].idxmax(),'profit']
print(f"Maximum profit (h2o): {amlprof}")

# Let's calculate accuracy using this decision threshold
amlpredictions = (y_pred_prob_aml >= amlprob).astype(int)
amlaccuracy = np.mean(amlpredictions == y_val)
print(f"Accuracy (h2o): {amlaccuracy}")








