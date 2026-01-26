# type: ignore
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Download the house price data
dropbox_url = "https://www.dropbox.com/scl/fi/7py0wnil5obkr16bwuwfw/WAHousePrice.xlsx?rlkey=s8riinvpub4n1dfg9cv9ypzhh&dl=1"
response = requests.get(dropbox_url)
data = pd.read_excel(BytesIO(response.content))
print(data.head())

# Flag to save plots
dpl = True   # Change to False to show plots (True saves to file)

# Scatter plot of price vs. size
plt.scatter(data['sqft_living'], data['price'])
plt.xlabel('Square Feet')
plt.ylabel('Price')
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size.png'))
    plt.close()
else:
    plt.show() 

# Some data cleaning
data = data[(data['bedrooms'] != 0) & (data['bathrooms'] != 0)]
data = data[data['price'] >= 50000]
data = data[data['price'] <= 10000000]

# Drop variables we definitely won't use
data = data.drop(columns=['date','street','country'])

# Split the data into training and validation sets
train_data, validation_data = train_test_split(data, test_size=1000, random_state=726)

# Let's plot price vs square footage in the training data
plt.scatter(train_data['sqft_living'], train_data['price'])
plt.xlabel("sqft_living")
plt.ylabel("price")
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size train.png'))
    plt.close()
else:
    plt.show() 

# For illustration purposes, create a variable which is square footage plus noise
np.random.seed(42)
train_data['sqft_noise'] = train_data['sqft_living'] + np.random.normal(0, 10, len(train_data))
validation_data['sqft_noise'] = validation_data['sqft_living'] + np.random.normal(0, 10, len(validation_data))

# Let's make sure adding noise doesn't change the relationship between price and square footage
plt.scatter(train_data['sqft_noise'], train_data['price'])
plt.xlabel("sqft_noise")
plt.ylabel("price")
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size noise.png'))
    plt.close()
else:
    plt.show() 

# Sample mean as prediction rule
ytrmean = np.mean(train_data['price'])
print('Sample mean of price: {m:=.2f}'.format(m=ytrmean))

# Mean squared error
mse_mean = np.mean((train_data['price'] - ytrmean)**2)
print('MSE of sample mean: {m:=.2f}'.format(m=mse_mean))

# Root mean squared error
rmse_mean = np.sqrt(mse_mean)
print('RMSE of sample mean: {m:=.2f}'.format(m=rmse_mean))

# Mean absolute error
mae_mean = np.mean(np.abs(train_data['price'] - ytrmean))
print('MAE of sample mean: {m:=.2f}'.format(m=mae_mean))

# R^2
r2_mean = 0

# Scatter plot with mean line drawn on
plt.scatter(train_data['sqft_noise'], train_data['price'])
plt.plot([min(train_data['sqft_noise']), max(train_data['sqft_noise'])], [ytrmean, ytrmean], color='red')
plt.xlabel("sqft_noise")
plt.ylabel("price")
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size noise mean.png'))
    plt.close()
else:
    plt.show()


# Linear model
model_lm = LinearRegression()
model_lm.fit(train_data[['sqft_noise']], train_data['price'])

# Fitted values
train_data['lm_price_fitted'] = model_lm.predict(train_data[['sqft_noise']])

# MSE
mse_lm = np.mean((train_data['price'] - train_data['lm_price_fitted'])**2)
print('MSE of linear model: {m:=.2f}'.format(m=mse_lm))

# RMSE
rmse_lm = np.sqrt(mse_lm)
print('RMSE of linear model: {m:=.2f}'.format(m=rmse_lm))

# MAE
mae_lm = np.mean(np.abs(train_data['price'] - train_data['lm_price_fitted']))
print('MAE of linear model: {m:=.2f}'.format(m=mae_lm))

#R^2
r2_lm= model_lm.score(train_data[['sqft_noise']], train_data['price'])
print("Linear model R^2 in training data: %f" % r2_lm)

# Scatter plot with linear model line drawn on
sorted_data = train_data.sort_values(by='sqft_noise')
plt.scatter(train_data['sqft_noise'], train_data['price'])
plt.plot(sorted_data['sqft_noise'], sorted_data['lm_price_fitted'], color='red')
plt.xlabel("sqft_noise")
plt.ylabel("price")
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size noise linear.png'))
    plt.close()
else:
    plt.show()

# Fit a tree model in the training data with four leaves
model_t4 = DecisionTreeRegressor(max_leaf_nodes=4)
model_t4.fit(train_data[['sqft_noise']], train_data['price'])

# Fitted values
train_data['tree_price_fitted'] = model_t4.predict(train_data[['sqft_noise']])

# Mean squared error
mse_tree = np.mean((train_data['price'] - train_data['tree_price_fitted'])**2)
print('MSE of tree model with 4 leaves: {m:=.2f}'.format(m=mse_tree))

# RMSE
rmse_tree = np.sqrt(mse_tree)
print('RMSE of tree model with 4 leaves: {m:=.2f}'.format(m=rmse_tree))

# MAE 
mae_tree = np.mean(np.abs(train_data['price'] - train_data['tree_price_fitted']))
print('MAE of tree model with 4 leaves: {m:=.2f}'.format(m=mae_tree))

# R^2
r2_tree = model_t4.score(train_data[['sqft_noise']], train_data['price'])
print("Tree model with 4 leaves R^2 in training data: %f" % r2_tree)

# Scatter plot with tree fit drawn on
sorted_data = train_data.sort_values(by='sqft_noise')
plt.scatter(train_data['sqft_noise'], train_data['price'])
plt.plot(sorted_data['sqft_noise'], sorted_data['tree_price_fitted'], color='red')
plt.xlabel("sqft_noise")
plt.ylabel("price")
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size noise tree4.png'))
    plt.close()
else:
    plt.show()
    
# Fit a KNN model with 80 neighbors
model_knn = KNeighborsRegressor(n_neighbors=50)
model_knn.fit(train_data[['sqft_noise']], train_data['price'])

# Fitted values
train_data['knn_price_fitted'] = model_knn.predict(train_data[['sqft_noise']])

# Mean squared error
mse_knn = np.mean((train_data['price'] - train_data['knn_price_fitted'])**2)
print('MSE of KNN model with 50 neighbors: {m:=.2f}'.format(m=mse_knn))

# RMSE
rmse_knn = np.sqrt(mse_knn)
print('RMSE of KNN model with 50 neighbors: {m:=.2f}'.format(m=rmse_knn))

# MAE 
mae_knn = np.mean(np.abs(train_data['price'] - train_data['knn_price_fitted']))
print('MAE of KNN model with 50 neighbors: {m:=.2f}'.format(m=mae_knn))

# R^2
r2_knn = model_knn.score(train_data[['sqft_noise']], train_data['price'])
print("KNN model with 50 neighbors R^2 in training data: %f" % r2_knn)

# Plot estimated model vs original data
sorted_data = train_data.sort_values(by='sqft_noise')
plt.scatter(train_data['sqft_noise'], train_data['price'])
plt.plot(sorted_data['sqft_noise'], sorted_data['knn_price_fitted'], color='red')
plt.xlabel("sqft_noise")
plt.ylabel("price")
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size noise knn50.png'))
    plt.close()
else:
    plt.show()

# Fit a tree model in the training data with one observation per leaf
model = DecisionTreeRegressor(min_samples_leaf=1)
model.fit(train_data[['sqft_noise']], train_data['price'])

# Fitted values
train_data['perfect_price_fitted'] = model.predict(train_data[['sqft_noise']])

# Mean squared error
mse_perfect = np.mean((train_data['price'] - train_data['perfect_price_fitted'])**2)
print('MSE of BIG tree model: {m:=.2f}'.format(m=mse_perfect))

# RMSE
rmse_perfect = np.sqrt(mse_perfect)
print('RMSE of BIG tree model: {m:=.2f}'.format(m=rmse_perfect))

# MAE
mae_perfect = np.mean(np.abs(train_data['price'] - train_data['perfect_price_fitted']))
print('MAE of BIG tree model: {m:=.2f}'.format(m=mae_perfect))

# Calculate in-sample R^2
r2_perfect = model.score(train_data[['sqft_noise']], train_data['price'])
print("R^2 performance of the model in training data: %f" % r2_perfect)

# Plot estimated tree model vs original data
sorted_data = train_data.sort_values(by='sqft_noise')
plt.scatter(train_data['sqft_noise'], train_data['price'])
plt.plot(sorted_data['sqft_noise'], sorted_data['perfect_price_fitted'], color='red')
plt.xlabel("sqft_noise")
plt.ylabel("price")
if dpl: 
    plt.savefig(os.path.join(script_dir, 'price vs size noise perfect.png'))
    plt.close()
else:
    plt.show()


# Model performance in the validation data

# First form predictions for each model in the validation data
validation_data['mean_predict'] = ytrmean
validation_data['lm_predict'] = model_lm.predict(validation_data[['sqft_noise']])
validation_data['tree_predict'] = model_t4.predict(validation_data[['sqft_noise']])
validation_data['knn_predict'] = model_knn.predict(validation_data[['sqft_noise']])
validation_data['perfect_predict'] = model.predict(validation_data[['sqft_noise']])

# MSEs
val_mse_mean = np.mean((validation_data['price'] - validation_data['mean_predict'])**2)
val_mse_lm = np.mean((validation_data['price'] - validation_data['lm_predict'])**2)
val_mse_tree = np.mean((validation_data['price'] - validation_data['tree_predict'])**2)
val_mse_knn = np.mean((validation_data['price'] - validation_data['knn_predict'])**2)
val_mse_perfect = np.mean((validation_data['price'] - validation_data['perfect_predict'])**2)

# RMSEs
val_rmse_mean = np.sqrt(val_mse_mean)
val_rmse_lm = np.sqrt(val_mse_lm)
val_rmse_tree = np.sqrt(val_mse_tree)
val_rmse_knn = np.sqrt(val_mse_knn)
val_rmse_perfect = np.sqrt(val_mse_perfect)

# MAEs
val_mae_mean = np.mean(np.abs(validation_data['price'] - validation_data['mean_predict']))
val_mae_lm = np.mean(np.abs(validation_data['price'] - validation_data['lm_predict']))
val_mae_tree = np.mean(np.abs(validation_data['price'] - validation_data['tree_predict']))
val_mae_knn = np.mean(np.abs(validation_data['price'] - validation_data['knn_predict']))
val_mae_perfect = np.mean(np.abs(validation_data['price'] - validation_data['perfect_predict']))  

# OOS R^2
val_r2_mean = 1 - val_mse_mean/val_mse_mean
val_r2_lm = 1 - val_mse_lm/val_mse_mean
val_r2_tree = 1 - val_mse_tree/val_mse_mean
val_r2_knn = 1 - val_mse_knn/val_mse_mean
val_r2_perfect = 1 - val_mse_perfect/val_mse_mean

# Data frame comparing in-sample to out-of-sample fit measures
df_compare = pd.DataFrame({
    'Model': ['Mean', 'Linear', 'Tree4', 'KNN50', 'Perfect'],
    'In MSE': [mse_mean, mse_lm, mse_tree, mse_knn, mse_perfect],
    'In RMSE': [rmse_mean, rmse_lm, rmse_tree, rmse_knn, rmse_perfect],
    'In MAE': [mae_mean, mae_lm, mae_tree, mae_knn, mae_perfect],
    'In R^2': [r2_mean, r2_lm, r2_tree, r2_knn, r2_perfect],
    'Out MSE': [val_mse_mean, val_mse_lm, val_mse_tree, val_mse_knn, val_mse_perfect],
    'Out RMSE': [val_rmse_mean, val_rmse_lm, val_rmse_tree, val_rmse_knn, val_rmse_perfect],
    'Out MAE': [val_mae_mean, val_mae_lm, val_mae_tree, val_mae_knn, val_mae_perfect],
    'Out R^2': [val_r2_mean, val_r2_lm, val_r2_tree, val_r2_knn, val_r2_perfect]
})

# Print data frame
print(df_compare)

# Let's repeat the exercise using 5 fold cross-validation in the training data
# and compare cross-validation to what we get in the validation data

# First, let's create a function that will perform 5 fold cross-validation
# and return the mean of the cross-validation results

class MeanRegressor:
    def fit(self, X, y):
        self.mean_ = np.mean(y)
    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)
    def score(self, X, y):
        # R^2 for mean model is always 0
        return 0

model_mean = MeanRegressor()

def cross_validate(model, data, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=726)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    for train_index, val_index in kf.split(data):
        tr_data = data.iloc[train_index]
        val_data = data.iloc[val_index].copy()  
        model.fit(tr_data[['sqft_noise']], tr_data['price'])
        val_data['predict'] = model.predict(val_data[['sqft_noise']])   
        mse_scores.append(np.mean((val_data['price'] - val_data['predict'])**2))
        rmse_scores.append(np.sqrt(np.mean((val_data['price'] - val_data['predict'])**2)))
        mae_scores.append(np.mean(np.abs(val_data['price'] - val_data['predict'])))
        r2_scores.append(1-np.mean((val_data['price'] - val_data['predict'])**2)/np.mean((val_data['price'] - np.mean(tr_data['price']))**2))
    return mse_scores, rmse_scores, mae_scores, r2_scores

# Now let's use the function to cross-validate the models and print the fold scores
mse_mean_cv, rmse_mean_cv, mae_mean_cv, r2_mean_cv = cross_validate(model_mean, train_data)
print('MeanRegressor fold scores:')
print('MSE:', mse_mean_cv)
print('RMSE:', rmse_mean_cv)
print('MAE:', mae_mean_cv)
print('R2:', r2_mean_cv)

mse_lm_cv, rmse_lm_cv, mae_lm_cv, r2_lm_cv = cross_validate(model_lm, train_data)
print('LinearRegression fold scores:')
print('MSE:', mse_lm_cv)
print('RMSE:', rmse_lm_cv)
print('MAE:', mae_lm_cv)
print('R2:', r2_lm_cv)

mse_tree_cv, rmse_tree_cv, mae_tree_cv, r2_tree_cv = cross_validate(model_t4, train_data)
print('Tree4 fold scores:')
print('MSE:', mse_tree_cv)
print('RMSE:', rmse_tree_cv)
print('MAE:', mae_tree_cv)
print('R2:', r2_tree_cv)

mse_knn_cv, rmse_knn_cv, mae_knn_cv, r2_knn_cv = cross_validate(model_knn, train_data)
print('KNN50 fold scores:')
print('MSE:', mse_knn_cv)
print('RMSE:', rmse_knn_cv)
print('MAE:', mae_knn_cv)
print('R2:', r2_knn_cv)

mse_perfect_cv, rmse_perfect_cv, mae_perfect_cv, r2_perfect_cv = cross_validate(model, train_data)
print('PerfectTree fold scores:')
print('MSE:', mse_perfect_cv)
print('RMSE:', rmse_perfect_cv)
print('MAE:', mae_perfect_cv)
print('R2:', r2_perfect_cv)

# Now let's compare the cross-validation results to the validation data results
# Aggregate the means for df_compare_cv as before
df_compare_cv = pd.DataFrame({
    'Model': ['Mean', 'Linear', 'Tree4', 'KNN50', 'Perfect'],
    'CV MSE': [np.mean(mse_mean_cv), np.mean(mse_lm_cv), np.mean(mse_tree_cv), np.mean(mse_knn_cv), np.mean(mse_perfect_cv)],
    'CV RMSE': [np.mean(rmse_mean_cv), np.mean(rmse_lm_cv), np.mean(rmse_tree_cv), np.mean(rmse_knn_cv), np.mean(rmse_perfect_cv)],
    'CV MAE': [np.mean(mae_mean_cv), np.mean(mae_lm_cv), np.mean(mae_tree_cv), np.mean(mae_knn_cv), np.mean(mae_perfect_cv)],
    'CV R^2': [np.mean(r2_mean_cv), np.mean(r2_lm_cv), np.mean(r2_tree_cv), np.mean(r2_knn_cv), np.mean(r2_perfect_cv)],
    'Out MSE': [val_mse_mean, val_mse_lm, val_mse_tree, val_mse_knn, val_mse_perfect],
    'Out RMSE': [val_rmse_mean, val_rmse_lm, val_rmse_tree, val_rmse_knn, val_rmse_perfect],
    'Out MAE': [val_mae_mean, val_mae_lm, val_mae_tree, val_mae_knn, val_mae_perfect],
    'Out R^2': [val_r2_mean, val_r2_lm, val_r2_tree, val_r2_knn, val_r2_perfect]
})

print(df_compare_cv)
