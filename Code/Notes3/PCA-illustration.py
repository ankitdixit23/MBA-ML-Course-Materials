# Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import requests
from io import BytesIO
import os

# switch to turn off plots (Switch to True to turn on plots)
dpl = False

# Get the directory where this script is located for saving plots
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
dropbox_url = "https://www.dropbox.com/scl/fi/rr3h3vnj6n6uswm1yctzu/AAPL_NVDA_2024.xlsx?rlkey=narlopxxwb6mtxalgus7zntt6&dl=1"
response = requests.get(dropbox_url)
retdata = pd.read_excel(BytesIO(response.content))
print(retdata.head())

# For simplicity, rename the columns to AAPL and NVDA
retdata.columns = ["Date", "AAPL", "NVDA"]

# Just keep the return columns
retdata = retdata[["AAPL","NVDA"]]
print(retdata.head())

# Extract the two principal components
pca = PCA(n_components=2)
ret_pca = pca.fit_transform(retdata)
ret_pca_directions = pca.components_
ret_pca_var = pca.explained_variance_
ret_pca_variance = pca.explained_variance_ratio_

# For seeing loadings
loadings = pd.DataFrame(ret_pca_directions, columns=retdata.columns, 
                        index=["PC1","PC2"])

# Add explained variance to loadings data frame
loadings["Fraction Explained Variance"] = ret_pca_variance

print(loadings)

ret_mean = retdata.mean()

# Plot PC directions on scatter plot of AAPL vs NVDA
plt.scatter(retdata['AAPL'], retdata['NVDA'])
# Set limits to have same range for AAPL and NVDA
plt.xlim([-0.35,0.4])
plt.ylim([-0.35,0.4])
# Add principal component directions to scatterplot
# Scale according to component standard deviation
for i in range(2):
    v = 3*np.sqrt(ret_pca_var[i]) * ret_pca_directions[i]
    x_vals = [ret_mean.iloc[0] - v[0], ret_mean.iloc[0] + v[0]]
    y_vals = [ret_mean.iloc[1] - v[1], ret_mean.iloc[1] + v[1]]
    plt.plot(x_vals, y_vals, linewidth=2, label=f"PC{i+1}")

plt.legend()
plt.xlabel('AAPL')
plt.ylabel('NVDA')
plt.title('AAPL vs NVDA with PCs')
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "return_PCA.png"), dpi=300)
    plt.close()

# Repeat with standardized data
retdata_std = StandardScaler().fit_transform(retdata)
ret_pca_std = pca.fit_transform(retdata_std)
ret_pca_directions_std = pca.components_
ret_pca_var_std = pca.explained_variance_
ret_pca_variance_std = pca.explained_variance_ratio_

# For seeing loadings
loadings_std = pd.DataFrame(ret_pca_directions_std, columns=retdata.columns, 
                        index=["PC1","PC2"])

# Add explained variance to loadings data frame
loadings_std["Fraction Explained Variance"] = ret_pca_variance_std

print(loadings_std)

# Plot PC directions on scatter plot of standardized AAPL vs NVDA
plt.scatter(retdata_std[:,0], retdata_std[:,1])
# Add principal component directions to scatterplot
# Scale according to component standard deviation
for i in range(2):
    v = 3*np.sqrt(ret_pca_var_std[i])*ret_pca_directions_std[i]
    x_vals = [-v[0], v[0]]
    y_vals = [-v[1], v[1]]
    plt.plot(x_vals, y_vals, linewidth=2, label=f"PC{i+1}")
# Set xlim and ylim to +/-3
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend()
plt.xlabel('AAPL')
plt.ylabel('NVDA')
plt.title('AAPL vs NVDA (Standardized) with PCs')
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "return_PCA_standardized.png"), dpi=300)
    plt.close()





