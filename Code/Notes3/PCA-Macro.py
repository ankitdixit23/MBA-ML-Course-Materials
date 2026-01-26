# Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
dropbox_url = "https://www.dropbox.com/scl/fi/qqw8frg6y7m84y2kn9n34/FRED-monthly-2025-07.csv?rlkey=d7nhg4s30eeh08e9rw0r6775g&dl=1"
response = requests.get(dropbox_url)
fred = pd.read_csv(BytesIO(response.content), encoding='latin1')
print(fred.head())

# Separate out date column
date = fred['Date']
fred = fred.drop(columns=['Date'])

# Find complete cases - i.e. find non-missing entries (coded as NaN)
complete = fred.dropna()

# Show number of variables and sample size in original and complete data
print(fred.shape)
print(complete.shape)

# Standardize the data before pca
scaler = StandardScaler()
scaler.fit(complete)
scaled_data = scaler.transform(complete)

# Do PCA
pca = PCA(n_components = 40)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data) 

eigs = pca.explained_variance_
pve = pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, 41), eigs, marker='o')
plt.title("Scree Plot (Variance)")
plt.xlabel("Number of Components (k)")
plt.ylabel("Variance")
plt.grid(True, alpha=0.3)
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "macro_scree_plot.png"))
    plt.close()

# Cumulative PVE
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, 41), cum_pve, marker='o')
plt.title("Cumulative Proportion of Variance Explained (PVE)")
plt.xlabel("Number of Components (k)")
plt.ylabel("Cumulative PVE")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "macro_cum_pve.png"))
    plt.close()

# Let's consider 6 components
components_6 = pca.components_[:6, :]
eigs_6 = eigs[:6]
# Loadings are component directions scaled by standard deviations
# Often used for interpretation
loadings = components_6.T * np.sqrt(eigs_6)

# Get variable names to help interpretation
varnames = list(complete.columns)

loadings_df = pd.DataFrame(loadings, index=varnames, columns=[f"PC{i+1}" for i in range(6)])

# Show top 10 by |loading| per component
top_rows = []
for i in range(6):
    col = loadings_df.iloc[:, i]
    top_vars = col.abs().sort_values(ascending=False).head(10).index.tolist()
    for rank, var in enumerate(top_vars, start=1):
        top_rows.append({"PC": f"PC{i+1}", "Rank": rank, "Variable": var, "Loading": loadings_df.loc[var, f"PC{i+1}"]})
top_df = pd.DataFrame(top_rows)
print(top_df)

# Heatmap
n_features = loadings_df.shape[0]
height = min(2 + 0.2 * n_features, 24)
plt.figure(figsize=(1 + 0.35 * 6, height))
plt.imshow(loadings_df.values, aspect='auto', interpolation='nearest')
plt.colorbar(label="Loading")
plt.title("PCA Loadings Heatmap")
plt.yticks(ticks=np.arange(n_features), labels=varnames, fontsize=6 )
plt.xticks(ticks=np.arange(6), labels=[f"PC{i+1}" for i in range(6)], fontsize=8, rotation = 90 )
plt.xlabel("Components")
plt.ylabel("Variables")
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "macro_loadings_heatmap.png"), bbox_inches='tight')
    plt.close()

# Heatmap without variable names
n_features = loadings_df.shape[0]
plt.imshow(loadings_df.values, aspect='auto', interpolation='nearest')
plt.colorbar(label="Loading")
plt.title("PCA Loadings Heatmap")
plt.xticks(ticks=np.arange(6), labels=[f"PC{i+1}" for i in range(6)], fontsize=8, rotation = 90 )
plt.xlabel("Components")
plt.ylabel("Variables")
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "macro_loadings_heatmap_nolabels.png"), bbox_inches='tight')
    plt.close()

# NBER recession dates during period covered by complete cases    
recessions = [
    ('2001-03', '2001-11'),
    ('2007-12', '2009-06'),
    ('2020-02', '2020-04')
]

# Convert to datetime
recessions = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in recessions]

date = pd.to_datetime(date)

# Time series plot of the extracted factors
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(6):
    ax.plot(date.loc[complete.index], x_pca[:, i], label=f'Factor {i+1}')
ax.legend()
ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

for start, end in recessions:
    ax.axvspan(start, end, color='gray', alpha=0.3)
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "macro_factors_time_series.png"), bbox_inches='tight')
    plt.close()
