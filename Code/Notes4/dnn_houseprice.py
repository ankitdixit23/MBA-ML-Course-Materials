# Run using Python 3.11 | TensorFlow/Keras 2.15

import os
import json
import math
import argparse
from io import BytesIO
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from formulaic import model_matrix
from sklearn.linear_model import LinearRegression

import requests

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input, callbacks

RNG = 726
TEST_SIZE_ROWS = 1000

tf.keras.utils.set_random_seed(RNG)

DROPBOX_URL = (
    "https://www.dropbox.com/scl/fi/7py0wnil5obkr16bwuwfw/WAHousePrice.xlsx?rlkey="
    "s8riinvpub4n1dfg9cv9ypzhh&dl=1"
)

# ---------------------------
# Config of variables
# ---------------------------
NUMERIC_VARS = [
    'bathrooms','bedrooms','sqft_living','sqft_lot','sqft_above','floors',
    'view','condition','yr_built','yr_renovated','sqft_basement'
]
BINARY_VARS  = ['waterfront','renovated_flag','basement_flag']
CAT_CITY = 'city'
CAT_STATEZIP = 'statezip'

# ---------------------------
# Utils
# ---------------------------

def _make_ohe():
    # sklearn >=1.2 uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def plot_history(history, title, outdir, start_epoch=20):
    hist = history.history
    epochs = range(start_epoch, len(hist['loss']) + 1)
    plt.figure()
    plt.plot(epochs, hist['loss'][start_epoch-1:], label='train')
    if 'val_loss' in hist:
        plt.plot(epochs, hist['val_loss'][start_epoch-1:], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log price)')
    plt.title(title)
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{title.replace(' ', '_').lower()}_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")


def rmse_levels(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r2_levels(y_train_log, y_test_log, y_pred_log):
    """R^2 in levels using the TRAIN mean level as the baseline."""
    y_train_mean_level = np.exp(y_train_log).mean()
    y_test_level = np.exp(y_test_log)
    y_pred_level = np.exp(y_pred_log)
    baseline_mse = mean_squared_error(y_test_level, np.full_like(y_test_level, y_train_mean_level))
    model_mse = mean_squared_error(y_test_level, y_pred_level)
    return 1.0 - (model_mse / baseline_mse)

# ---------------------------
# Data loading & cleaning
# ---------------------------

def load_data() -> pd.DataFrame:
    print("Downloading data ...")
    resp = requests.get(DROPBOX_URL)
    resp.raise_for_status()
    df = pd.read_excel(BytesIO(resp.content))
    print(f"Loaded shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Basic sanity filters
    df = df[df['price'] >= 50000]
    df = df[df['price'] <= 10000000]
    df = df[df['bathrooms'] > 0]
    df = df[df['bedrooms'] > 0]

    # Drop irrelevants
    drop_cols = [c for c in ['date', 'street', 'country'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Targets / transforms
    df['log_price'] = np.log(df['price'])
    for col in ['sqft_living', 'sqft_lot', 'sqft_above']:
        if col in df.columns:
            df[col] = np.log(df[col])

    # Engineered flags
    if 'yr_renovated' in df.columns:
        df['renovated_flag'] = np.where(df['yr_renovated'] == 0, 0, 1)
    else:
        df['renovated_flag'] = 0

    if 'sqft_basement' in df.columns:
        df['basement_flag'] = np.where(df['sqft_basement'] == 0, 0, 1)
        mask = df["sqft_basement"] > 0
        df.loc[mask, "sqft_basement"] = np.log(df.loc[mask, "sqft_basement"])
    else:
        df['basement_flag'] = 0

    df = df.reset_index(drop=True)
    print(f"After cleaning: {df.shape}")
    return df

# ---------------------------
# Splitting & preprocessing 
# ---------------------------

def split_indices(n_rows: int, test_size_rows: int, rng: int):
    idx = np.arange(n_rows)
    train_idx, test_idx = train_test_split(idx, test_size=test_size_rows, random_state=rng)
    return train_idx, test_idx


def fit_preprocessor_on_train(df_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in NUMERIC_VARS if c in df_train.columns]
    bin_cols     = [c for c in BINARY_VARS if c in df_train.columns]
    cat_cols     = [c for c in [CAT_CITY, CAT_STATEZIP] if c in df_train.columns]

    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=True, with_std=True), numeric_cols),
            ('bin', 'passthrough', bin_cols),
            ('cat', _make_ohe(), cat_cols),
        ],
        remainder='drop'
    )
    pre.fit(df_train)
    return pre


def transform_flat(pre: ColumnTransformer, df_part: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    X = pre.transform(df_part)

    # Names
    num_names = pre.transformers_[0][2]
    bin_names = pre.transformers_[1][2]
    ohe: OneHotEncoder = pre.named_transformers_['cat']
    cat_cols = pre.transformers_[2][2]
    ohe_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = list(num_names) + list(bin_names) + ohe_names

    # Index groups for branched slicing
    city_idx = [i for i, n in enumerate(ohe_names) if n.startswith(f"{CAT_CITY}_")]
    state_idx = [i for i, n in enumerate(ohe_names) if n.startswith(f"{CAT_STATEZIP}_")]
    offset = len(num_names) + len(bin_names)
    index_groups = {
        'numerics_and_bin': list(range(offset)),
        'city_dummies': [offset + i for i in city_idx],
        'statezip_dummies': [offset + i for i in state_idx]
    }

    return X.astype('float32'), feature_names, index_groups


def split_branched_from_flat(X: np.ndarray, index_groups: Dict[str, List[int]]):
    main_idx = index_groups['numerics_and_bin']
    city_idx = index_groups['city_dummies']
    state_idx = index_groups['statezip_dummies']
    X_main = X[:, main_idx] if len(main_idx)>0 else np.empty((len(X),0))
    X_city = X[:, city_idx] if len(city_idx)>0 else np.empty((len(X),0))
    X_state= X[:, state_idx] if len(state_idx)>0 else np.empty((len(X),0))
    return X_main.astype('float32'), X_city.astype('float32'), X_state.astype('float32')

# ---------------------------
# Models
# ---------------------------

def build_wide_shallow(input_dim: int, width=512, l2=None, dropout=None) -> tf.keras.Model:
    inputs = Input(shape=(input_dim,), name="flat_input")
    x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(l2) if l2 else None)(inputs)
    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="log_price")(x)
    model = Model(inputs, outputs, name="wide_shallow")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


def build_deep_narrow(input_dim: int, depth=6, width=32, l2=1e-4, dropout=None) -> tf.keras.Model:
    inputs = Input(shape=(input_dim,), name="flat_input")
    x = inputs
    for i in range(depth):
        x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(l2), name=f"d{i+1}")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="log_price")(x)
    model = Model(inputs, outputs, name="deep_narrow")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


def build_branched(input_dims: Dict[str,int], l2=1e-4) -> tf.keras.Model:
    inp_main = Input(shape=(input_dims['main'],), name="inp_main")
    inp_city = Input(shape=(input_dims['city'],), name="inp_city")
    inp_state= Input(shape=(input_dims['state'],), name="inp_state")

    c = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l2), name="city_h1")(inp_city)
    c = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l2), name="city_h2")(c)

    s = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l2), name="state_h1")(inp_state)
    s = layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l2), name="state_h2")(s)

    m = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2), name="main_h1")(inp_main)
    m = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2), name="main_h2")(m)

    z = layers.Concatenate(name="concat")([m, c, s])
    z = layers.Dense(128, activation='relu', name="post_concat_h1")(z)
    z = layers.Dense(128, activation='relu', name="post_concat_h2")(z)
    out = layers.Dense(1, name="log_price")(z)

    model = Model([inp_main, inp_city, inp_state], out, name="branched_towers")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model

# ---------------------------
# Training / evaluation
# ---------------------------

def train_with_val(model, X_train, y_train, validation_split=0.2, early_stop=False, patience=10):
    cbs = []
    if early_stop:
        cbs.append(callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=256,
        validation_split=validation_split,
        verbose=0,
        callbacks=cbs,
    )
    return history


def evaluate_model(model, X_test, y_test, y_train, label="model"):
    if hasattr(model, 'predict') and 'verbose' in model.predict.__code__.co_varnames:
        ypred = model.predict(X_test, verbose=0).reshape(-1)
    else:
        ypred = model.predict(X_test).reshape(-1)
        
    rmse = rmse_levels(y_test, ypred)
    r2 = r2_levels(y_train, y_test, ypred)
    print(f"[{label}] Test RMSE (levels): {rmse:,.0f}   R² (levels): {r2:0.3f}")
    return ypred, rmse, r2

# ---------------------------
# permutation importance, bias, activations
# ---------------------------

def permutation_importance(predict_fn: Callable[[np.ndarray], np.ndarray], X_test, y_test, feature_names, repeats=3):
    rng = np.random.default_rng(RNG)
    base_rmse = rmse_levels(y_test, predict_fn(X_test))
    importances = []
    for j in range(X_test.shape[1]):
        deltas = []
        for _ in range(repeats):
            Xp = X_test.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, j] = Xp[perm, j]
            rmse_p = rmse_levels(y_test, predict_fn(Xp))
            deltas.append(rmse_p - base_rmse)
        importances.append((feature_names[j], float(np.mean(deltas))))
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances


def bias_report(predict_fn: Callable[[np.ndarray], np.ndarray], X_test, y_test, basement_test, outdir, label="branched"):
    yhat_log = predict_fn(X_test)
    yhat_lev = np.exp(yhat_log)
    ytrue_lev = np.exp(y_test)

    df = pd.DataFrame({
        "y_true": ytrue_lev,
        "y_pred": yhat_lev,
        "resid": ytrue_lev - yhat_lev,
        "basement_flag": basement_test
    })

    report = []
    for g in [0,1]:
        dfg = df[df['basement_flag']==g]
        mae = dfg['resid'].abs().mean()
        rmse = math.sqrt((dfg['resid']**2).mean())
        n_obs = len(dfg)
        report.append((g, n_obs, dfg['y_pred'].mean(), dfg['y_true'].mean(), 
                       dfg['y_pred'].median(), dfg['y_true'].median(), mae, rmse, 
                       dfg['resid'].mean()))
    rep_df = pd.DataFrame(report, columns=['basement_flag','n','mean_pred','mean_true',
                                           'median_pred','median_true',
                                           'MAE','RMSE','mean_resid'])
    print("\n=== Bias/Disparity (by basement_flag) ===")
    print(rep_df)

    rep_df.to_csv(os.path.join(outdir, "bias_report.csv"), index=False)
    print(f"[Saved] {os.path.join(outdir, 'bias_report.csv')}")

    os.makedirs(outdir, exist_ok=True)
    for col in ['n','mean_pred', 'mean_true', 'median_pred', 'median_true', 
                'MAE', 'RMSE', 'mean_resid']:
        plt.figure()
        plt.bar(rep_df['basement_flag'].astype(str), rep_df[col].values)
        plt.xlabel('basement_flag')
        plt.ylabel(col)
        plt.title(f'{col} by basement_flag')
        f = os.path.join(outdir, f"bias_{col}.png")
        plt.savefig(f, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {f}")


def get_layer_outputs(model: tf.keras.Model, layer_names: List[str], X):
    submodels = [Model(model.inputs, model.get_layer(n).output) for n in layer_names]
    outs = [sm.predict(X, verbose=0) for sm in submodels]
    return {n: o for n, o in zip(layer_names, outs)}


def compare_activations_by_group(act_dict_main, groups, outdir, title_prefix="activations"):
    os.makedirs(outdir, exist_ok=True)
    uniq = np.unique(groups)
    for lname, A in act_dict_main.items():
        means = []
        for g in uniq:
            means.append(A[groups==g].mean(axis=0))
        means = np.vstack(means)  # G x H
        for unit in range(means.shape[1]):
            plt.figure()
            for i, g in enumerate(uniq):
                plt.bar(str(g), means[i, unit])
            plt.xlabel('basement_flag')
            plt.ylabel('mean activation')
            plt.title(f'{title_prefix}: {lname} unit {unit}')
            f = os.path.join(outdir, f"{title_prefix}_{lname}_unit{unit}.png")
            plt.savefig(f, dpi=120, bbox_inches='tight')
            plt.close()


def heatmap_activations_by_group(
    act_dict_main,
    groups,
    outdir,
    layer_order=None,
    title_prefix="activations",
    group_labels=None,
    normalize="layer_zscore",      # None | "layer_minmax" | "layer_zscore"
    cmap="viridis",
    vmin=None, vmax=None,
    show_colorbar=True,
    figure_dpi=150,
    difference_panel=True          # if exactly two groups, also save a |mean(g1)-mean(g2)| heatmap
):
    os.makedirs(outdir, exist_ok=True)
    groups = np.asarray(groups)
    uniq = np.array(sorted(np.unique(groups), key=lambda x: str(x)))

    # Consistent layer order
    if layer_order is None:
        layer_order = list(act_dict_main.keys())
    else:
        layer_order = [L for L in layer_order if L in act_dict_main]

    # --- Compute mean activations per group and layer ---
    # For each layer, build a matrix G x H (groups x hidden_units)
    layer_mean_by_group = {}
    for lname in layer_order:
        A = np.asarray(act_dict_main[lname])
        H = A.shape[1]
        M = np.zeros((len(uniq), H))
        for i, g in enumerate(uniq):
            idx = (groups == g)
            if idx.sum() == 0:
                M[i, :] = np.nan
            else:
                M[i, :] = A[idx].mean(axis=0)
        layer_mean_by_group[lname] = M  # shape G x H

    # --- Build concatenated matrices per group: rows=1 (this group), columns=all units across layers ---
    # Also keep bookkeeping to draw vertical separators and layer labels
    layer_col_spans = []  # list of (start_col, end_col_exclusive, layer_name)
    total_units = 0
    for lname in layer_order:
        G, H = layer_mean_by_group[lname].shape
        layer_col_spans.append((total_units, total_units + H, lname))
        total_units += H

    # Function to optionally normalize *within each layer across groups & units of that layer’s group means*
    def normalize_per_layer(concat_vec_by_group):
        if normalize is None:
            return concat_vec_by_group, None, None

        # Build layer-wise stats from all groups' means so scaling is comparable across groups
        layer_stats = []
        for (c0, c1, lname) in layer_col_spans:
            block = np.vstack([vec[c0:c1] for vec in concat_vec_by_group])  # G x H
            if normalize == "layer_minmax":
                minv = np.nanmin(block)
                maxv = np.nanmax(block)
                rng = maxv - minv if maxv > minv else 1.0
                layer_stats.append(("minmax", minv, rng))
            elif normalize == "layer_zscore":
                mu = np.nanmean(block)
                sd = np.nanstd(block, ddof=0)
                if not np.isfinite(sd) or sd == 0:
                    sd = 1.0
                layer_stats.append(("zscore", mu, sd))
            else:
                raise ValueError("normalize must be None|'layer_minmax'|'layer_zscore'")

        # Apply normalization layer-by-layer
        norm_concat = []
        for g_idx, vec in enumerate(concat_vec_by_group):
            vec = vec.copy()
            for (c0, c1, _), stat in zip(layer_col_spans, layer_stats):
                mode, a, b = stat
                if mode == "minmax":
                    vec[c0:c1] = (vec[c0:c1] - a) / b
                else:  # zscore
                    vec[c0:c1] = (vec[c0:c1] - a) / b
            norm_concat.append(vec)
        return norm_concat, layer_stats, normalize

    # Build concatenated vectors per group
    concat_by_group = []
    for i, g in enumerate(uniq):
        cols = []
        for lname in layer_order:
            cols.append(layer_mean_by_group[lname][i])  # length H_l
        concat_by_group.append(np.concatenate(cols, axis=0))  # shape total_units,

    # Normalize (optional)
    concat_by_group, layer_stats, norm_mode = normalize_per_layer(concat_by_group)

    # --- Plot one heatmap per group (1 x total_units) but draw as (rows=1) image for readability ---
    # For each group, we’ll plot a 2D array with shape (1, total_units)
    def _draw_heatmap(vec, g, fname_suffix, suptitle):
        arr = vec.reshape(1, -1)
        fig_h = 2.2  # short strip per group
        fig_w = max(8, min(18, total_units / 60 + 6))  # scale width by number of units
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=figure_dpi)
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks([])
        ax.set_xlabel("Units concatenated across layers")
        # Vertical layer separators & labels
        for (c0, c1, lname) in layer_col_spans:
            ax.axvline(c0 - 0.5, color='w', lw=1)
            # center label above
            mid = (c0 + c1 - 1) / 2.0
            ax.text(mid, -0.6, lname, ha='center', va='bottom', fontsize=9, rotation=0, color='w',
                    bbox=dict(facecolor='black', alpha=0.25, pad=2, edgecolor='none'))
        # Rightmost border line
        ax.axvline(total_units - 0.5, color='w', lw=1)
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
            if norm_mode == "layer_zscore":
                cbar.set_label("Z-score (within layer)")
            elif norm_mode == "layer_minmax":
                cbar.set_label("Min‑max (within layer)")
            else:
                cbar.set_label("Mean activation")
        ax.set_title(suptitle, fontsize=11)
        fname = os.path.join(outdir, f"{title_prefix}_{fname_suffix}.png")
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        return fname

    # Render per-group heatmaps
    saved = []
    for i, g in enumerate(uniq):
        glabel = group_labels.get(g, str(g)) if isinstance(group_labels, dict) else str(g)
        suptitle = f"{title_prefix} · group={glabel}"
        fname = _draw_heatmap(concat_by_group[i], g, f"group_{glabel}", suptitle)
        saved.append(fname)

    # Optional: difference heatmap if exactly two groups
    if difference_panel and len(uniq) == 2:
        diff_vec = np.abs(concat_by_group[0] - concat_by_group[1])
        suptitle = f"{title_prefix} · |mean({uniq[0]}) - mean({uniq[1]})|"
        fname = _draw_heatmap(diff_vec, "diff", "group_diff_abs", suptitle)
        saved.append(fname)

    # Also: add a compact multi‑row figure with all groups stacked (easier side‑by‑side scanning)
    fig_h = 1.6 * len(uniq) + 0.6
    fig_w = max(8, min(18, total_units / 60 + 6))
    fig, axes = plt.subplots(len(uniq), 1, figsize=(fig_w, fig_h), dpi=figure_dpi, sharex=True)
    if len(uniq) == 1:
        axes = [axes]
    for ax, i in zip(axes, range(len(uniq))):
        arr = concat_by_group[i].reshape(1, -1)
        im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks([])
        glabel = group_labels.get(uniq[i], str(uniq[i])) if isinstance(group_labels, dict) else str(uniq[i])
        ax.set_ylabel(glabel, rotation=0, ha='right', va='center', labelpad=8)
        for (c0, c1, lname) in layer_col_spans:
            ax.axvline(c0 - 0.5, color='w', lw=1)
        ax.axvline(total_units - 0.5, color='w', lw=1)

    axes[-1].set_xlabel("Units concatenated across layers")
    # Layer labels on top across the stacked figure
    top_ax = axes[0]
    for (c0, c1, lname) in layer_col_spans:
        mid = (c0 + c1 - 1) / 2.0
        top_ax.text(mid, -0.8, lname, ha='center', va='bottom', fontsize=9, color='w',
                    bbox=dict(facecolor='black', alpha=0.25, pad=2, edgecolor='none'))
    if show_colorbar:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
        if norm_mode == "layer_zscore":
            cbar.set_label("Z-score (within layer)")
        elif norm_mode == "layer_minmax":
            cbar.set_label("Min‑max (within layer)")
        else:
            cbar.set_label("Mean activation")
    fig.suptitle(f"{title_prefix} · per‑group heatmaps", y=1.02, fontsize=12)
    fname_all = os.path.join(outdir, f"{title_prefix}_all_groups_stacked.png")
    plt.savefig(fname_all, bbox_inches='tight')
    plt.close(fig)

    return saved + [fname_all]


# ---------------------------
# Main
# ---------------------------

def main(outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)

    # 1) Load & clean
    df = load_data()
    df = clean_data(df)

    # 2) Train/test row indices
    y_all = df['log_price'].values.astype('float32')
    train_idx, test_idx = split_indices(len(df), TEST_SIZE_ROWS, RNG)

    # For group fairness charts
    basement_all = df['basement_flag'].values.astype(int)
    basement_test = basement_all[test_idx]

    # 3) Preprocess (fit on TRAIN only; transform both)
    pre = fit_preprocessor_on_train(df.iloc[train_idx])

    Xtr, feat_names, idx_groups = transform_flat(pre, df.iloc[train_idx])
    Xte, _,         _           = transform_flat(pre, df.iloc[test_idx])
    ytr, yte = y_all[train_idx], y_all[test_idx]

    print(f"Xtr: {Xtr.shape}  Xte: {Xte.shape}")

    # Also build branched arrays from flat (no refits)
    Xm_tr, Xc_tr, Xs_tr = split_branched_from_flat(Xtr, idx_groups)
    Xm_te, Xc_te, Xs_te = split_branched_from_flat(Xte, idx_groups)

    # 4) Train models
    # linear regression
    base = ("price ~ poly(bathrooms, degree=2, raw=True) + poly(bedrooms, degree=2, raw=True)"
             " + poly(sqft_living, degree=2, raw=True) + poly(sqft_lot, degree=2, raw=True) + poly(floors, degree=2, raw=True) "
            "+ waterfront + poly(view, degree=2, raw=True) + poly(condition, degree=2, raw=True) + "
            "poly(yr_built, degree=2, raw=True) + poly(yr_renovated, degree=2, raw=True) + C(renovated_flag) + C(basement_flag)"
            " + sqft_living:(bedrooms+bathrooms) + C(city) + C(statezip)")

    expy, X = model_matrix(base, df)
    y = df['log_price']
    Xtr_lr, Xte_lr = X.iloc[train_idx], X.iloc[test_idx]
    ytr_lr, yte_lr = y.iloc[train_idx], y.iloc[test_idx]
    lr = LinearRegression()
    lr.fit(Xtr_lr, ytr_lr)
    ypred_lr = lr.predict(Xte_lr)
    ypred_lr, rmse_lr, r2_lr = evaluate_model(lr, Xte_lr, yte_lr, ytr_lr, label="LinearRegression")

    # DNNs
    wide = build_wide_shallow(Xtr.shape[1], l2=None, dropout=None)
    print(wide.summary())
    h_wide = train_with_val(wide, Xtr, ytr, validation_split=0.2, early_stop=False)
    plot_history(h_wide, "WideShallow", outdir)
    ypred_wide, rmse_wide, r2_wide = evaluate_model(wide, Xte, yte, ytr, 
                                                    label="WideShallow")

    widedr = build_wide_shallow(Xtr.shape[1], l2=None, dropout=0.3)
    h_widedr = train_with_val(widedr, Xtr, ytr, validation_split=0.2, early_stop=False)
    plot_history(h_widedr, "WideShallow_Dropout", outdir)
    ypred_widedr, rmse_widedr, r2_widedr = evaluate_model(widedr, Xte, yte, ytr, 
                                                          label="WideShallow - Dropout")

    widel2 = build_wide_shallow(Xtr.shape[1], l2=1e-4, dropout=None)
    h_widel2 = train_with_val(widel2, Xtr, ytr, validation_split=0.2, early_stop=False)
    plot_history(h_widel2, "WideShallow_l2", outdir)
    ypred_widel2, rmse_widel2, r2_widel2 = evaluate_model(widel2, Xte, yte, ytr, 
                                                          label="WideShallow - l2")

    widees = build_wide_shallow(Xtr.shape[1], l2=None, dropout=None)
    h_widees = train_with_val(widees, Xtr, ytr, validation_split=0.2, early_stop=True, patience=10)
    plot_history(h_widees, "WideShallow_es", outdir)
    ypred_widees, rmse_widees, r2_widees = evaluate_model(widees, Xte, yte, ytr, 
                                                          label="WideShallow - EarlyStop")

    deep = build_deep_narrow(Xtr.shape[1], depth=6, width=32, l2=None)
    print(deep.summary())
    h_deep = train_with_val(deep, Xtr, ytr, validation_split=0.2, early_stop=False)
    plot_history(h_deep, "DeepNarrow", outdir)
    ypred_deep, rmse_deep, r2_deep = evaluate_model(deep, Xte, yte, ytr, 
                                                    label="DeepNarrow")

    deepes = build_deep_narrow(Xtr.shape[1], depth=6, width=32, l2=None)
    h_deepes = train_with_val(deepes, Xtr, ytr, validation_split=0.2, early_stop=True, patience=10)
    plot_history(h_deepes, "DeepNarrow_EarlyStop", outdir)
    ypred_deepes, rmse_deepes, r2_deepes = evaluate_model(deepes, Xte, yte, ytr, 
                                                          label="DeepNarrow - EarlyStop")

    branched = build_branched({'main': Xm_tr.shape[1], 'city': Xc_tr.shape[1], 'state': Xs_tr.shape[1]}, l2=None)
    print(branched.summary())
    h_branch = train_with_val(branched, [Xm_tr, Xc_tr, Xs_tr], ytr, validation_split=0.2, early_stop=False)
    plot_history(h_branch, "Branched", outdir)
    ypred_branch, rmse_branch, r2_branch = evaluate_model(branched, [Xm_te, Xc_te, Xs_te], 
                                                          yte, ytr, label="Branched")

    branchedl2 = build_branched({'main': Xm_tr.shape[1], 'city': Xc_tr.shape[1], 'state': Xs_tr.shape[1]}, l2=1e-4)
    h_branchl2 = train_with_val(branchedl2, [Xm_tr, Xc_tr, Xs_tr], ytr, validation_split=0.2, early_stop=False)
    plot_history(h_branchl2, "Branched_l2", outdir)
    ypred_branchl2, rmse_branchl2, r2_branchl2 = evaluate_model(branchedl2, [Xm_te, Xc_te, Xs_te], 
                                                                yte, ytr, label="Branched - l2")

    branchedes = build_branched({'main': Xm_tr.shape[1], 'city': Xc_tr.shape[1], 'state': Xs_tr.shape[1]}, l2=None)
    h_branches = train_with_val(branchedes, [Xm_tr, Xc_tr, Xs_tr], ytr, validation_split=0.2, 
                                early_stop=True, patience=10)
    plot_history(h_branches, "Branched_EarlyStop", outdir)
    ypred_branches, rmse_branches, r2_branches = evaluate_model(branchedes, [Xm_te, Xc_te, Xs_te], 
                                                                yte, ytr, label="Branched - EarlyStop")

    deepw = build_deep_narrow(Xtr.shape[1], depth=6, width=128, l2=None)
    print(deepw.summary())
    h_deepw = train_with_val(deepw, Xtr, ytr, validation_split=0.2, early_stop=False)
    plot_history(h_deepw, "Deep", outdir)
    ypred_deepw, rmse_deepw, r2_deepw = evaluate_model(deepw, Xte, yte, ytr, label="Deep")

    deepwes = build_deep_narrow(Xtr.shape[1], depth=6, width=128, l2=None, dropout=None)
    h_deepwes = train_with_val(deepwes, Xtr, ytr, validation_split=0.2, early_stop=True, patience=10)
    plot_history(h_deepwes, "Deep_EarlyStop", outdir)
    ypred_deepwes, rmse_deepwes, r2_deepwes = evaluate_model(deepwes, Xte, yte, ytr, label="Deep - EarlyStop")


    # 5) Interpretation using branched model but flat inputs
    def predict_branched_on_flat(Xflat: np.ndarray) -> np.ndarray:
        main_idx = idx_groups['numerics_and_bin']
        city_idx = idx_groups['city_dummies']
        state_idx = idx_groups['statezip_dummies']
        X_main = Xflat[:, main_idx]
        X_city = Xflat[:, city_idx] if len(city_idx)>0 else np.empty((len(Xflat),0), dtype=Xflat.dtype)
        X_state = Xflat[:, state_idx] if len(state_idx)>0 else np.empty((len(Xflat),0), dtype=Xflat.dtype)
        return branched.predict([X_main, X_city, X_state], verbose=0).reshape(-1)

    rng = np.random.default_rng(RNG)
    tr_sub_idx = rng.choice(Xtr.shape[0], size=min(2000, Xtr.shape[0]), replace=False)
    Xref = Xtr[tr_sub_idx].copy()

    # PDP: sqft_living
    idx_sqft = feat_names.index('sqft_living')
    xs = np.linspace(Xtr[:, idx_sqft].min(), Xtr[:, idx_sqft].max(), 25)
    pdp_sqft = []
    for v in xs:
        Xtmp = Xref.copy()
        Xtmp[:, idx_sqft] = v
        yhat = predict_branched_on_flat(Xtmp)
        pdp_sqft.append(np.exp(yhat).mean())
    plt.figure()
    plt.plot(xs, pdp_sqft)
    plt.xlabel('scaled sqft_living')
    plt.ylabel('Predicted price (levels)')
    plt.title('PDP: sqft_living (branched model)')
    plt.twinx()
    plt.hist(Xref[:, idx_sqft], bins=30, color='gray', alpha=0.7)
    plt.ylabel('Count (histogram)')
    f = os.path.join(outdir, "pdp_sqft_living.png")
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {f}")

    # PDP: bedrooms
    idx_bed = feat_names.index('bedrooms')
    xs = np.linspace(Xtr[:, idx_bed].min(), Xtr[:, idx_bed].max(), 9)
    pdp_bed = []
    for v in xs:
        Xtmp = Xref.copy()
        Xtmp[:, idx_bed] = v
        yhat = predict_branched_on_flat(Xtmp)
        pdp_bed.append(np.exp(yhat).mean())
    plt.figure()
    plt.plot(xs, pdp_bed)
    plt.xlabel('scaled bedrooms')
    plt.ylabel('Predicted price (levels)')
    plt.title('PDP: bedrooms (branched model)')
    plt.twinx()
    plt.hist(Xref[:, idx_bed], bins=30, color='gray', alpha=0.7)
    plt.ylabel('Count (histogram)')
    f = os.path.join(outdir, "pdp_bedrooms.png")
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {f}")

    # Get histogram for square footage and number of bedrooms
    plt.figure()
    plt.hist(df['sqft_living'], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('sqft_living (log scale)')
    plt.ylabel('Count')
    plt.title('Histogram of sqft_living')
    f = os.path.join(outdir, "hist_sqft_living.png")
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {f}")

    plt.figure()
    plt.hist(df['bedrooms'], bins=np.arange(df['bedrooms'].min(), df['bedrooms'].max()+2)-0.5, color='salmon', edgecolor='black')
    plt.xlabel('bedrooms')
    plt.ylabel('Count')
    plt.title('Histogram of bedrooms')
    f = os.path.join(outdir, "hist_bedrooms.png")
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {f}")

    # Plot average price per unique value of number of bedrooms
    avg_price_by_bedrooms = df.groupby('bedrooms')['price'].mean()
    plt.figure()
    plt.bar(avg_price_by_bedrooms.index.astype(str), avg_price_by_bedrooms.values, color='teal', edgecolor='black')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Average Price')
    plt.title('Average Price by Number of Bedrooms')
    f = os.path.join(outdir, "avg_price_by_bedrooms.png")
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {f}")


    # Permutation importance (branched predictor on flat Xte)
    print("\nComputing permutation importance on test set (branched model)...")
    imp = permutation_importance(lambda X: predict_branched_on_flat(X), Xte, yte, feat_names, repeats=3)
    imp_df = pd.DataFrame(imp, columns=['feature','ΔRMSE'])
    print(imp_df.head(15))

    plt.figure()
    topk = min(15, len(imp_df))
    plt.barh(np.arange(topk), imp_df['ΔRMSE'].iloc[:topk][::-1].values)
    plt.yticks(np.arange(topk), imp_df['feature'].iloc[:topk][::-1].values)
    plt.xlabel('Increase in RMSE (levels) when permuted')
    plt.title('Permutation importance (branched model)')
    f = os.path.join(outdir, "perm_importance.png")
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {f}")

    # 6) Bias diagnostics on branched model
    bias_report(
        predict_fn=lambda X: predict_branched_on_flat(X),
        X_test=Xte,
        y_test=yte,
        basement_test=basement_test,
        outdir=outdir,
        label="branched",
    )

    # 7) Activation audit (branched)
    layer_names = ["city_h2", "state_h2", "post_concat_h1"]
    acts = get_layer_outputs(branched, layer_names, [Xm_te, Xc_te, Xs_te])
    compare_activations_by_group(acts, basement_test, outdir, title_prefix="branched")

    # 8) Heatmap activations
    layer_names = ["city_h1", "city_h2", "state_h1", "state_h2", "main_h1", "main_h2",
                   "post_concat_h1", "post_concat_h2"]
    acts = get_layer_outputs(branched, layer_names, [Xm_te, Xc_te, Xs_te])
    saved_files = heatmap_activations_by_group(
        act_dict_main=acts,                
        groups=basement_test,              
        outdir=outdir,
        layer_order=["city_h1", "city_h2", "state_h1", "state_h2", "main_h1", "main_h2",
                   "post_concat_h1", "post_concat_h2"],  
        title_prefix="branched",
        group_labels={0: "no_basement", 1: "basement"},
        normalize="layer_zscore",          
        cmap="viridis",
        difference_panel=True
    )
    print("\n".join(saved_files)) 

    # 9) Save headline metrics
    metrics = {
        "LinearRegression":          {"RMSE_levels": float(rmse_lr),       "R2_levels": float(r2_lr)},
        "WideShallow":               {"RMSE_levels": float(rmse_wide),     "R2_levels": float(r2_wide)},
        "WideShallow_Dropout":       {"RMSE_levels": float(rmse_widedr),   "R2_levels": float(r2_widedr)},
        "WideShallow_l2":            {"RMSE_levels": float(rmse_widel2),   "R2_levels": float(r2_widel2)},
        "WideShallow_EarlyStop":     {"RMSE_levels": float(rmse_widees),   "R2_levels": float(r2_widees)},
        "DeepNarrow":                {"RMSE_levels": float(rmse_deep),     "R2_levels": float(r2_deep)},
        "DeepNarrow_EarlyStop":      {"RMSE_levels": float(rmse_deepes),   "R2_levels": float(r2_deepes)},
        "Deep":                      {"RMSE_levels": float(rmse_deepw),    "R2_levels": float(r2_deepw)},
        "Deep_EarlyStop":            {"RMSE_levels": float(rmse_deepwes),  "R2_levels": float(r2_deepwes)},
        "Branched":                  {"RMSE_levels": float(rmse_branch),   "R2_levels": float(r2_branch)},
        "Branched_l2":               {"RMSE_levels": float(rmse_branchl2), "R2_levels": float(r2_branchl2)},
        "Branched_EarlyStop":        {"RMSE_levels": float(rmse_branches), "R2_levels": float(r2_branches)},
    }

    # Save JSON
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save a CSV for convenience
    pd.DataFrame.from_dict(metrics, orient="index")\
      .reset_index().rename(columns={"index": "Model"})\
      .to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    print("\nSaved metrics.json, metrics.csv, and all figures in:", outdir)


# Ensure outputs are saved relative to script location
if '__file__' in globals():
    BASEDIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASEDIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASEDIR, "outputs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(outdir=args.outdir)
