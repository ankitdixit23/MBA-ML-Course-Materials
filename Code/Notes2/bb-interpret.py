# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import clone
from joblib import Parallel, delayed
import seaborn as sns
import requests
import os
from io import BytesIO

# switch to turn off plots (Switch to True to turn on plots)
dpl = True

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

# Function to compute drop column variable importance
def drop_column_importance(
    baseline_score,
    X_train,
    y_train,
    X_test,
    y_test,
    metric_fn,
    cloneable_model,
    use_proba=False,
    columns=None,
    n_jobs=-1,
    verbose=0
):
    """
    Calculate feature importance by dropping each column and measuring the performance drop.

    Parameters:
    - baseline_score: Baseline metric score with all features.
    - X_train, y_train: Training data.
    - X_test, y_test: Testing data.
    - metric_fn: Scoring function (e.g., accuracy_score, f1_score, roc_auc_score).
    - cloneable_model: A scikit-learn Pipeline or estimator. Must be cloneable.
    - use_proba: Use predict_proba instead of predict for scoring.
    - n_jobs: Parallel jobs for computing feature importances.
    - verbose: Verbosity level for joblib.

    Returns:
    - List of (feature_name, importance) tuples, sorted by importance.
    """

    def remove_column_from_transformer(transformer, dropped_col):
        new_transformers = []
        for name, trans, cols in transformer.transformers:
            if cols == 'drop' or trans == 'drop':
                new_transformers.append((name, 'drop', cols))
            elif cols == 'passthrough' or trans == 'passthrough':
                new_transformers.append((name, trans, cols))
            else:
                new_cols = [col for col in cols if col != dropped_col]
                if new_cols:
                    new_transformers.append((name, trans, new_cols))
                else:
                    new_transformers.append((name, 'drop', []))
        return ColumnTransformer(new_transformers, remainder=transformer.remainder)

    def compute_importance(col):
        # Drop the column
        X_train_dropped = X_train.drop(columns=[col])
        X_test_dropped = X_test.drop(columns=[col])

        # Clone the model and fit
        model = clone(cloneable_model)

        # If it's a pipeline and uses a ColumnTransformer, rebuild the preprocessor
        if isinstance(model, Pipeline):
            if hasattr(model.named_steps['preprocessor'], 'transformers'):
                preproc = model.named_steps['preprocessor']
                new_preproc = remove_column_from_transformer(preproc, col)
                model.steps[0] = ('preprocessor', new_preproc)

        model.fit(X_train_dropped, y_train)

        # Predict and score
        preds = model.predict_proba(X_test_dropped)[:, 1] if use_proba else model.predict(X_test_dropped)
        dropped_score = metric_fn(y_test, preds)

        # Compute percentage drop in performance
        importance = (baseline_score - dropped_score) / baseline_score
        return col, importance, dropped_score

    if columns is None:
        columns = X_train.columns    

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_importance)(col) for col in columns
    )

    return sorted(results, key=lambda x: x[1], reverse=True)

# sklearn's built in partial dependence doesn't work well with categorical variables.
# This function computes ICE curves for a single variable, allowing for both numeric and categorical variables.
def compute_ice_curves(
    pipeline,
    X,
    variable,
    grid=None,
    grid_resolution=100,
    is_categorical=False,
    use_proba=False,
    plot=True
):
    """
    Compute ICE curves (and their average) for a single variable.

    Parameters:
    - pipeline: Fitted sklearn pipeline with a preprocessor and classifier.
    - X: DataFrame of input features (before preprocessing).
    - variable: Name of the variable to vary (string).
    - grid: Optional list of values at which to evaluate ICE.
    - grid_resolution: Number of points for numeric ICE grid (default: 100).
    - is_categorical: If True, treat variable as categorical.
    - use_proba: If True, use predict_proba[:,1] instead of predict().
    - plot: If True, display ICE plot (line plot for numeric, boxplot for categorical).

    Returns:
    - ice_df: DataFrame of shape (n_samples, len(grid)) with ICE values.
    - grid_values: Array of evaluated grid points.
    """
    X = X.copy()
    n_samples = X.shape[0]

    # Determine evaluation grid
    if grid is not None:
        grid_values = np.array(grid)
    elif is_categorical:
        grid_values = np.sort(X[variable].dropna().unique())
    else:
        unique_vals = np.sort(X[variable].dropna().unique())
        if len(unique_vals) <= grid_resolution:
            grid_values = unique_vals
        else:
            grid_values = np.linspace(X[variable].min(), X[variable].max(), grid_resolution)

    # Compute ICE matrix
    ice_matrix = []
    for val in grid_values:
        X_temp = X.copy()
        X_temp[variable] = val
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_temp)
        model = pipeline.named_steps['classifier']
        preds = model.predict_proba(X_transformed)[:, 1] if use_proba else model.predict(X_transformed)
        ice_matrix.append(preds)

    ice_array = np.stack(ice_matrix, axis=1)  # shape: (n_samples, n_grid)
    ice_df = pd.DataFrame(ice_array, columns=grid_values)

    # Plotting
    if plot:
        if is_categorical:
            fig, ax = plt.subplots(figsize=(8, 6))
            ice_long = pd.melt(ice_df.assign(sample=np.arange(n_samples)), id_vars='sample')
            ice_long.columns = ['sample', 'category', 'prediction']
            sns.boxplot(x='category', y='prediction', data=ice_long, ax=ax)
            ax.set_title(f'ICE Box Plot for Categorical Variable: {variable}')
            ax.set_xlabel(variable)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            for row in ice_df.itertuples(index=False):
                ax.plot(grid_values, row, color='gray', alpha=0.2)
            ax.plot(grid_values, ice_array.mean(axis=0), color='red', lw=2, label='Average')
            ax.set_title(f'ICE Curves for {variable}')
            ax.set_xlabel(variable)
            ax.set_ylabel('Predicted Response')
            ax.legend()

        plt.tight_layout()
        plt.show()

    return ice_df, grid_values

# Estimate baseline model performance
numeric_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'never_contacted']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# From exercise showing validation performance, the best random forest we tried 
# used min_samples_leaf = 15 and n_estimators = 1000 (though most were very similar)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=1000, min_samples_leaf=15, random_state=94, n_jobs=-1))
])

# Compute baseline scores
pipeline.fit(X_train, y_train)
baseline_preds = pipeline.predict(X_val)
baseline_predprobs = pipeline.predict_proba(X_val)[:, 1]
baseline_acc = accuracy_score(y_val, baseline_preds)
baseline_auc = roc_auc_score(y_val, baseline_predprobs)
baseline_logloss = log_loss(y_val, baseline_predprobs)
print(f"Baseline Accuracy: {baseline_acc}")
print(f"Baseline AUC: {baseline_auc}")
print(f"Baseline Log Loss: {baseline_logloss}")

# Drop column importance (accuracy)
dc_acc = drop_column_importance(baseline_acc, X_train, y_train, X_val, y_val, accuracy_score, pipeline)

# Display the results
dc_acc_df = pd.DataFrame(dc_acc, columns=['Feature', 'Importance', 'DroppedScore'])
dc_acc_df = dc_acc_df.sort_values(by='Importance', ascending=False)

print(dc_acc_df)

# Plot the feature importances
if dpl:
    dc_acc_df.plot(kind='bar', x='Feature', y='Importance',
                   title='Drop Importance - Accuracy', legend=False)
    plt.show()

# Drop column importance (cross-entropy)
dc_ce = drop_column_importance(baseline_logloss, X_train, y_train, X_val, y_val, log_loss, pipeline, use_proba=True)

# Display the results
dc_ce_df = pd.DataFrame(dc_ce, columns=['Feature', 'Importance', 'DroppedScore'])
# Multiply importance by -1 because it is a loss
dc_ce_df['Importance'] = -1*dc_ce_df['Importance']
dc_ce_df = dc_ce_df.sort_values(by='Importance', ascending=False)

print(dc_ce_df)

# Plot the feature importances
if dpl:
    dc_ce_df.plot(kind='bar', x='Feature', y='Importance',
                   title='Drop Importance - Cross-Entropy', legend=False)
    plt.show()

# Drop column importance (auc)
dc_auc = drop_column_importance(baseline_auc, X_train, y_train, X_val, y_val, roc_auc_score, pipeline, use_proba=True)

# Display the results
dc_auc_df = pd.DataFrame(dc_auc, columns=['Feature', 'Importance', 'DroppedScore'])
dc_auc_df = dc_auc_df.sort_values(by='Importance', ascending=False)

print(dc_auc_df)

# Plot the feature importances
if dpl:
    dc_auc_df.plot(kind='bar', x='Feature', y='Importance',
                   title='Drop Importance - AUC', legend=False)
    plt.show()

# Now let's look at permutation importance (accuracy)
perm_acc = permutation_importance(
    estimator=pipeline,
    X=X_val,
    y=y_val,
    scoring='accuracy',  
    n_repeats=30,
    n_jobs=-1,
    random_state=94
)

perm_acc_df = pd.DataFrame({
    'Feature': X_val.columns,
    'Importance': perm_acc.importances_mean,
    'Std': perm_acc.importances_std
}).sort_values('Importance', ascending=False)
print(perm_acc_df)

# Plot the feature importances
if dpl:
    perm_acc_df.plot(kind='bar', x='Feature', y='Importance',
                   title='Permutation Importance - Accuracy', legend=False)
    plt.show()

# Now let's look at permutation importance (cross-entropy)
perm_ce = permutation_importance(
    estimator=pipeline,
    X=X_val,
    y=y_val,
    scoring='neg_log_loss', 
    n_repeats=30,
    n_jobs=-1,
    random_state=94
)

perm_ce_df = pd.DataFrame({
    'Feature': X_val.columns,
    'Importance': perm_ce.importances_mean,
    'Std': perm_ce.importances_std
}).sort_values('Importance', ascending=False)
print(perm_ce_df)

# Plot the feature importances
if dpl:
    perm_ce_df.plot(kind='bar', x='Feature', y='Importance',
                   title='Permutation Importance - Cross entropy', legend=False)
    plt.show()

# Now let's look at permutation importance (auc)
perm_auc = permutation_importance(
    estimator=pipeline,
    X=X_val,
    y=y_val,
    scoring='roc_auc',  
    n_repeats=30,
    n_jobs=-1,
    random_state=94
)

perm_auc_df = pd.DataFrame({
    'Feature': X_val.columns,
    'Importance': perm_auc.importances_mean,
    'Std': perm_auc.importances_std
}).sort_values('Importance', ascending=False)
print(perm_auc_df)

# Plot the feature importances
if dpl:
    perm_auc_df.plot(kind='bar', x='Feature', y='Importance',
                     title='Permutation Importance - AUC', legend=False)
    plt.show()

# Finally let's look at a couple of partial dependence plots
pdp_eur = PartialDependenceDisplay.from_estimator(pipeline, X_val, [X_val.columns.get_loc('euribor3m')], kind='both')

if dpl:
    for ax in pdp_eur.axes_.ravel():
        lines = ax.get_lines()
        if lines:
            lines[-1].set_color('red')  
            lines[-1].set_linewidth(2.5)

    plt.show()

    # Euribor3m
    ice_df, grid = compute_ice_curves(pipeline, X_val, 'euribor3m', use_proba=True)

    # contact
    ice_cont_df, cont_vals = compute_ice_curves(pipeline, X_val, 'contact', is_categorical=True, use_proba=True)

    # job
    ice_cat_df, cat_vals = compute_ice_curves(pipeline, X_val, 'job', is_categorical=True, use_proba=True, plot=False)

    # Reshape ice_cat_df into long format
    n_samples = ice_cat_df.shape[0]
    ice_long = pd.melt(
        ice_cat_df.assign(sample=np.arange(n_samples)),
        id_vars='sample',
        var_name='category',
        value_name='prediction'
    )

    # Plot boxplot with rotated x-axis labels
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='category', y='prediction', data=ice_long, ax=ax)
    ax.set_title("ICE Box Plot for Categorical Variable")
    ax.set_xlabel("Category")
    ax.set_ylabel("Predicted Response")
    plt.xticks(rotation=90)  # <- rotates labels vertically
    plt.tight_layout()
    plt.show()

