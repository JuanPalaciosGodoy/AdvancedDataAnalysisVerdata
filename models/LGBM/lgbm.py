# lgbm_panel.py

import numpy as np
import pandas as pd
import polars as pl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import lightgbm as lgb

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import shap


# -------------------------------------------------------------------
# 1. DATA PREPARATION
# -------------------------------------------------------------------

def prepare_panel_for_lgbm(
    df_polars: pl.DataFrame,
    outcome: str,
    sector_cols=None,
    n_pcs: int = 3,
    max_lag: int = 12,
    use_population: bool = True,
):
    if sector_cols is None:
        sector_cols = [f"S0{i}" for i in range(1, 9)]

    # Ensure types and sort
    df_polars = (
        df_polars
        .with_columns([
            pl.col("DEPT_CODE").cast(pl.Int32),
            pl.col("YYYYMM").cast(pl.Date),
        ])
        .sort(["DEPT_CODE", "YYYYMM"])
    )

    df_pd = df_polars.to_pandas().reset_index(drop=True)
    df_pd["YYYYMM"] = pd.to_datetime(df_pd["YYYYMM"])

    # --- indices (keep them in meta for convenience, but don't feed to LGBM directly) ---
    dept_codes = np.sort(df_pd["DEPT_CODE"].unique())
    time_vals = np.sort(df_pd["YYYYMM"].unique())
    dept_to_idx = {d: i for i, d in enumerate(dept_codes)}
    time_to_idx = {t: i for i, t in enumerate(time_vals)}

    df_pd["dept_idx"] = df_pd["DEPT_CODE"].map(dept_to_idx).astype(int)
    df_pd["time_idx"] = df_pd["YYYYMM"].map(time_to_idx).astype(int)

    # --- PCA on sector variables (global) ---
    X_sectors = df_pd[sector_cols].values.astype(float)
    scaler_sect = StandardScaler()
    X_scaled = scaler_sect.fit_transform(X_sectors)

    n_pcs = min(n_pcs, X_scaled.shape[1])
    pca = PCA(n_components=n_pcs)
    X_pcs = pca.fit_transform(X_scaled)

    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    for j, col in enumerate(pc_cols):
        df_pd[col] = X_pcs[:, j]

    # --- POPULATION (not in PCA, but standardized) ---
    extra_feats = []
    if use_population and "POPULATION" in df_pd.columns:
        pop = df_pd["POPULATION"].astype(float).values.reshape(-1, 1)
        scaler_pop = StandardScaler()
        pop_std = scaler_pop.fit_transform(pop).ravel()
        df_pd["POP_STD"] = pop_std
        extra_feats.append("POP_STD")
    else:
        scaler_pop = None

    # --- Time features ---
    df_pd["month"] = df_pd["YYYYMM"].dt.month
    df_pd["year"] = df_pd["YYYYMM"].dt.year

    # --- Lag features of the outcome (per dept) ---
    df_pd = df_pd.sort_values(["DEPT_CODE", "YYYYMM"])
    for lag in [1, 12]:
        if lag > max_lag:
            continue
        df_pd[f"{outcome}_lag{lag}"] = (
            df_pd.groupby("DEPT_CODE")[outcome]
            .shift(lag)
            .astype(float)
        )

    lag_cols = [c for c in df_pd.columns if c.startswith(outcome + "_lag")]
    df_feat = df_pd.dropna(subset=lag_cols).copy()

    # Outcome (on count scale)
    y = df_feat[outcome].astype(float).values

    # ⚠️ Important: include DEPT_CODE itself as a feature (categorical)
    feature_cols = pc_cols + extra_feats + ["month", "year"] + lag_cols + ["DEPT_CODE"]
    X = df_feat[feature_cols].astype(float).values

    meta = {
        "dept_codes": dept_codes,
        "time_vals": time_vals,
        "dept_to_idx": dept_to_idx,
        "time_to_idx": time_to_idx,
        "feature_cols": feature_cols,
        "pc_cols": pc_cols,
        "sector_cols": sector_cols,
        "outcome": outcome,
        "scaler_sect": scaler_sect,
        "pca": pca,
        "scaler_pop": scaler_pop,
        # which feature index is the department categorical
        "dept_cat_index": feature_cols.index("DEPT_CODE"),
    }

    return df_feat, X, y, meta



# -------------------------------------------------------------------
# 2. FIT LGBM ON PANEL
# -------------------------------------------------------------------

def fit_lgbm_panel(
    df_feat: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    meta: dict,
    n_estimators: int = 1500,
    learning_rate: float = 0.03,
    max_depth: int = -1,
    num_leaves: int = 63,
    min_data_in_leaf: int = 30,
    val_frac: float = 0.2,
):
    """
    Fit a single LightGBM Poisson model on the whole panel.

    Returns
    -------
    model : lightgbm.Booster
    df_feat_sorted : DataFrame sorted by YYYYMM
    metrics : dict with RMSE/MAE/Poisson-NLL on validation
    """

    # find validation start month on a sorted view
    tmp_sorted = df_feat.sort_values("YYYYMM")
    unique_months = tmp_sorted["YYYYMM"].unique()
    n_months = len(unique_months)
    n_val_months = int(n_months * val_frac)
    val_start_month = unique_months[-n_val_months]

    # masks aligned with df_feat / X / y
    train_mask = df_feat["YYYYMM"] < val_start_month
    val_mask = ~train_mask

    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_val = X[val_mask.values]
    y_val = y[val_mask.values]

    df_feat_sorted = df_feat.sort_values(["YYYYMM", "DEPT_CODE"]).reset_index(drop=True)

    dept_cat_index = meta["dept_cat_index"]

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=[dept_cat_index],
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_set,
        categorical_feature=[dept_cat_index],
        free_raw_data=False,
    )

    params = {
        "objective": "poisson",
        "metric": "rmse",         # we care about RMSE on counts
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,         # light ridge can help stability
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100),
        ],
    )

    # Validation predictions on count scale (Poisson mean)
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    eps = 1e-8
    y_pred_val = np.clip(y_pred_val, eps, None)

    rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
    mae = np.mean(np.abs(y_val - y_pred_val))
    poisson_nll = np.mean(y_pred_val - y_val * np.log(y_pred_val))

    metrics = {
        "rmse_val": float(rmse),
        "mae_val": float(mae),
        "poisson_nll_val": float(poisson_nll),
    }

    return model, df_feat_sorted, metrics




def add_lgbm_residuals(
    model: lgb.Booster,
    df_feat: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    exposure_col: str = "POPULATION",
    use_log_rate_target: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of df_feat with y_pred (on count scale) and resid = y - y_pred.
    """

    exposure = df_feat[exposure_col].astype(float).values
    log_exposure = np.log(exposure + 1e-8)

    eta_hat = model.predict(X, num_iteration=model.best_iteration)

    if use_log_rate_target:
        # model output is log-rate; convert to expected counts
        y_pred = np.exp(eta_hat + log_exposure) - 1.0
    else:
        y_pred = eta_hat

    eps = 1e-8
    y_pred = np.clip(y_pred, eps, None)

    resid = y - y_pred

    df_out = df_feat.copy()
    df_out["y_pred"] = y_pred
    df_out["resid"] = resid
    return df_out




# -------------------------------------------------------------------
# 3. DIAGNOSTICS & GOF (similar to SARIMAX)
# -------------------------------------------------------------------

def lgbm_diagnostics(
    model: lgb.Booster,
    df_feat: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    lags: int = 24,
    alpha: float = 0.05,
    exposure_col: str = "POPULATION",
    use_log_rate_target: bool = True,
):
    """
    Diagnostics for LightGBM panel model.
    """

    df_plot = add_lgbm_residuals(
        model,
        df_feat,
        X,
        y,
        exposure_col=exposure_col,
        use_log_rate_target=use_log_rate_target,
    )
    resid = df_plot["resid"].values

    # residuals over time
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    df_plot.sort_values("YYYYMM").set_index("YYYYMM")["resid"].plot(ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("LightGBM: residuals over time")

    # ACF/PACF of monthly mean residuals
    resid_ts = (
        df_plot.groupby("YYYYMM")["resid"]
        .mean()
        .sort_index()
    )

    plot_acf(resid_ts, ax=axes[1], lags=lags)
    axes[1].set_title("LightGBM: residual ACF (mean over depts)")

    plot_pacf(resid_ts, ax=axes[2], lags=lags, method="ywm")
    axes[2].set_title("LightGBM: residual PACF (mean over depts)")

    sm.qqplot(resid, line="45", ax=axes[3])
    axes[3].set_title("LightGBM: residual QQ plot (Gaussian reference)")
    plt.tight_layout()
    plt.show()

    # GOF tests
    lb = acorr_ljungbox(resid_ts, lags=[lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])

    jb_stat, jb_pvalue, _, _ = jarque_bera(resid)

    print(f"LightGBM Ljung–Box (lag {lags}) on mean residuals: stat={lb_stat:.3f}, p={lb_pvalue:.3f}")
    print(f"LightGBM Jarque–Bera on residuals: stat={jb_stat:.3f}, p={jb_pvalue:.3f}")

    gof = {
        "lb_stat": lb_stat,
        "lb_pvalue": lb_pvalue,
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
    }

    return gof, df_plot


def lgbm_gof_panel_all_depts(
    panel_df_pred: pd.DataFrame,
    lags: int = 24,
    alpha: float = 0.05,
):
    """
    Run Ljung–Box and Jarque–Bera per department.
    `panel_df_pred` must contain columns: DEPT_CODE, YYYYMM, resid.
    """

    rows = []
    for dept in sorted(panel_df_pred["DEPT_CODE"].unique()):
        df_d = panel_df_pred[panel_df_pred["DEPT_CODE"] == dept].copy()
        df_d = df_d.sort_values("YYYYMM")
        resid = pd.Series(df_d["resid"].values, index=df_d["YYYYMM"])

        lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])

        jb_stat, jb_pvalue, _, _ = jarque_bera(resid)

        rows.append(
            dict(
                DEPT_CODE=dept,
                lb_stat=lb_stat,
                lb_pvalue=lb_pvalue,
                jb_stat=float(jb_stat),
                jb_pvalue=float(jb_pvalue),
                lb_flag=lb_pvalue < alpha,
                jb_flag=jb_pvalue < alpha,
            )
        )

    summary = pd.DataFrame(rows).sort_values("DEPT_CODE").reset_index(drop=True)
    return summary



# -------------------------------------------------------------------
# 4. SPATIAL RESIDUAL MAPS (analogous to SARIMAX & NB-CAR)
# -------------------------------------------------------------------

def build_lgbm_residuals_geodf(
    panel_df_pred: pd.DataFrame,
    geojson_path: str,
    dept_code_col_geo: str = "dept_code_hecho",
):
    """
    Aggregate sum/mean absolute residuals by department and join to geometry.
    """

    rows = []
    for dept, df_d in panel_df_pred.groupby("DEPT_CODE"):
        resid = df_d["resid"].values.astype(float)
        rows.append(
            {
                "DEPT_CODE": int(dept),
                "sum_abs_resid": np.abs(resid).sum(),
                "mean_abs_resid": np.abs(resid).mean(),
            }
        )
    df_resid = pd.DataFrame(rows)

    gdf_dept = gpd.read_file(geojson_path)
    gdf_dept[dept_code_col_geo] = gdf_dept[dept_code_col_geo].astype(int)

    gdf_resid = gdf_dept.merge(
        df_resid,
        left_on=dept_code_col_geo,
        right_on="DEPT_CODE",
        how="left",
    )

    gdf_resid = gpd.GeoDataFrame(gdf_resid, geometry="geometry", crs="EPSG:4326")
    return gdf_resid


def plot_lgbm_spatial_residuals(
    gdf_resid: gpd.GeoDataFrame,
    value_col: str = "sum_abs_resid",
    title: str = "LightGBM: sum of absolute residuals by department",
):
    vals = gdf_resid[value_col].values
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    fig = px.choropleth(
        gdf_resid,
        geojson=gdf_resid.geometry,
        locations=gdf_resid.index,
        color=value_col,
        hover_name="dept_name_hecho",
        range_color=(vmin, vmax),
        scope="south america",
        title=title,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()


# -------------------------------------------------------------------
# 5. ROLLING-ORIGIN TIME-SERIES CV (panel)
# -------------------------------------------------------------------

def lgbm_rolling_cv_panel(
    df_feat: pd.DataFrame,
    meta: dict,
    n_folds: int = 3,
    horizon_months: int = 6,
    n_estimators: int = 1500,
    learning_rate: float = 0.03,
    max_depth: int = -1,
    num_leaves: int = 31,
    min_data_in_leaf: int = 50,
    val_frac: float = 0.2,
):
    """
    Rolling-origin time-series CV for the LightGBM panel model.

    Parameters
    ----------
    df_feat : DataFrame
        Output of prepare_panel_for_lgbm (the flattened panel).
    meta : dict
        Output meta from prepare_panel_for_lgbm (must contain
        'feature_cols', 'outcome', 'time_vals').
    n_folds : int
        Number of rolling CV folds.
    horizon_months : int
        Number of months ahead to evaluate for each fold.
    (The remaining arguments are passed to LightGBM.)

    Returns
    -------
    cv_results : pd.DataFrame
        One row per fold with RMSE / MAE / Poisson NLL.
    """

    feature_cols = meta["feature_cols"]
    outcome = meta["outcome"]

    # sort consistently
    df_sorted = df_feat.sort_values(["YYYYMM", "DEPT_CODE"]).reset_index(drop=True)

    time_vals = np.sort(df_sorted["YYYYMM"].unique())
    n_time = len(time_vals)

    cut_indices = np.linspace(
        int(n_time * 0.4),
        n_time - horizon_months - 1,
        n_folds,
        dtype=int,
    )

    rows = []

    for fold, cut_idx in enumerate(cut_indices, start=1):
        cutoff_time = time_vals[cut_idx]
        print(f"\n=== LGBM CV fold {fold}/{n_folds}: train <= {cutoff_time.date()} ===")

        # training data: all months up to cutoff_time
        train_mask = df_sorted["YYYYMM"] <= cutoff_time

        # test months: next `horizon_months` after cutoff
        future_months = time_vals[cut_idx + 1 : cut_idx + 1 + horizon_months]
        test_mask = df_sorted["YYYYMM"].isin(future_months)

        df_train = df_sorted[train_mask].copy()
        df_test = df_sorted[test_mask].copy()

        if df_test.empty:
            print("  (no test data for this fold; skipping)")
            continue

        X_train = df_train[feature_cols].astype(float).values
        y_train = df_train[outcome].astype(float).values

        X_test = df_test[feature_cols].astype(float).values
        y_test = df_test[outcome].astype(float).values

        # --- fit LightGBM on training slice (log-rate target) ---
        model_fold, _, _ = fit_lgbm_panel(
            df_train,
            X_train,
            y_train,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            val_frac=val_frac,
            exposure_col=meta.get("exposure_col", "POPULATION"),
            use_log_rate_target=True,
        )

        # --- evaluate on test slice (OUT-OF-SAMPLE, count scale) ---
        exposure_test = df_test[meta.get("exposure_col", "POPULATION")].astype(float).values
        log_exposure_test = np.log(exposure_test + 1e-8)

        eta_test = model_fold.predict(X_test, num_iteration=model_fold.best_iteration)
        mu_test = np.exp(eta_test + log_exposure_test) - 1.0
        eps = 1e-8
        mu_test = np.clip(mu_test, eps, None)

        rmse = float(np.sqrt(np.mean((y_test - mu_test) ** 2)))
        mae = float(np.mean(np.abs(y_test - mu_test)))
        poisson_nll = float(np.mean(mu_test - y_test * np.log(mu_test)))


        rows.append(
            dict(
                model="LGBM",
                fold=fold,
                cutoff_time=cutoff_time,
                horizon_months=horizon_months,
                n_train=len(df_train),
                n_test=len(df_test),
                rmse=rmse,
                mae=mae,
                poisson_nll=poisson_nll,
            )
        )

        print(
            f"  Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, Poisson-NLL={poisson_nll:.3f}"
        )

    cv_results = pd.DataFrame(rows)
    return cv_results


def lgbm_shap_summary(
    model: lgb.Booster,
    df_feat: pd.DataFrame,
    meta: dict,
    max_samples: int = 2000,
    show_plots: bool = True,
):
    """
    Compute SHAP values for the LightGBM panel model and (optionally)
    show global summary plots.

    Parameters
    ----------
    model : lightgbm.Booster
    df_feat : DataFrame
        Same design matrix used to train the model (or a subset).
    meta : dict
        Output of prepare_panel_for_lgbm; must contain 'feature_cols'.
    max_samples : int
        Randomly subsample at most this many rows for SHAP (for speed).
    show_plots : bool
        If True, display shap.summary_plot (bar + beeswarm).

    Returns
    -------
    shap_values : np.ndarray
        SHAP values (n_samples, n_features).
    X_sample : pd.DataFrame
        Feature matrix used for SHAP computation.
    shap_importance : pd.DataFrame
        Columns: feature, mean_abs_shap; sorted descending.
    """

    feature_cols = meta["feature_cols"]
    X_all = df_feat[feature_cols].astype(float)

    if (max_samples is not None) and (len(X_all) > max_samples):
        X_sample = X_all.sample(max_samples, random_state=123)
    else:
        X_sample = X_all

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # mean |SHAP| per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_importance = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    if show_plots:

        # bar plot of importance
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_cols,
            plot_type="bar",
            show=True,
        )

        # beeswarm plot for full distribution
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_cols,
            show=True,
        )

    return shap_values, X_sample, shap_importance


def lgbm_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    interaction_feature: str | None = None,
):
    """
    Convenience wrapper to make a SHAP dependence plot for one feature.

    Parameters
    ----------
    shap_values : np.ndarray
        Output from lgbm_shap_summary.
    X_sample : DataFrame
        Same as returned by lgbm_shap_summary.
    feature : str
        Feature name to plot.
    interaction_feature : str or None
        Optional second feature for coloring points.
    """
    shap.dependence_plot(
        feature,
        shap_values,
        X_sample,
        interaction_index=interaction_feature,
        show=True,
    )
