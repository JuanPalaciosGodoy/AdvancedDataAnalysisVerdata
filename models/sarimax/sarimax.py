# sarimax_panel.py

import numpy as np
import pandas as pd
import polars as pl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

import geopandas as gpd
import plotly.express as px



def prepare_panel_for_sarimax(
    df_polars: pl.DataFrame,
    outcome: str,
    sector_cols=None,
    n_pcs: int = 3,
    apply_sqrt_transform: bool = True,
    sqrt_offset: float = 0.5,
):
    """
    Prepare panel data for SARIMAX with:
      - PCA on sectoral GDP variables (S01..S08),
      - standardized POPULATION as an additional exogenous variable
        (not included in the PCA),
      - optional square-root transform of the outcome.
    """

    if sector_cols is None:
        sector_cols = [f"S0{i}" for i in range(1, 9)]

    # Ensure proper types and sort
    df_polars = (
        df_polars
        .with_columns([
            pl.col("DEPT_CODE").cast(pl.Int32),
            pl.col("YYYYMM").cast(pl.Date),
        ])
        .sort(["DEPT_CODE", "YYYYMM"])
    )

    df_pd = df_polars.to_pandas().reset_index(drop=True)

    # --- Standardize sector variables across entire panel ---
    X_sectors = df_pd[sector_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sectors)

    # --- PCA on sectors only ---
    n_pcs = min(n_pcs, X_scaled.shape[1])
    pca = PCA(n_components=n_pcs)
    X_pcs = pca.fit_transform(X_scaled)  # shape (N_obs, n_pcs)

    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    for j, col in enumerate(pc_cols):
        df_pd[col] = X_pcs[:, j]

    # --- Standardize POPULATION (not part of PCA) ---
    if "POPULATION" not in df_pd.columns:
        raise KeyError("POPULATION column not found in df_polars / df_pd.")

    pop_raw = df_pd["POPULATION"].astype(float).values
    pop_mean = pop_raw.mean()
    pop_std = pop_raw.std()
    if pop_std == 0:
        pop_std = 1.0
    pop_stdzd = (pop_raw - pop_mean) / pop_std

    pop_col = "POPULATION_std"
    df_pd[pop_col] = pop_stdzd

    # --- Build per-department time series ---
    df_pd["YYYYMM"] = pd.to_datetime(df_pd["YYYYMM"])

    dept_codes = np.sort(df_pd["DEPT_CODE"].unique())
    time_vals = np.sort(df_pd["YYYYMM"].unique())

    series_dict = {}

    for dept in dept_codes:
        df_d = (
            df_pd[df_pd["DEPT_CODE"] == dept]
            .sort_values("YYYYMM")
            .set_index("YYYYMM")
        )

        # original target on count scale
        y_raw = df_d[outcome].astype(float)

        # square-root transform if requested
        if apply_sqrt_transform:
            y = np.sqrt(y_raw + sqrt_offset)
        else:
            y = y_raw.copy()

        # PCs + standardized POPULATION as exogenous variables
        exog = df_d[pc_cols + [pop_col]].astype(float)

        series_dict[dept] = {
            "y": y,            # transformed (or raw) series used in SARIMAX
            "exog": exog,
            "y_raw": y_raw,    # keep a copy for later interpretation
        }

    meta = {
        "dept_codes": dept_codes,
        "time_vals": time_vals,
        "scaler": scaler,      # for sectors
        "pca": pca,
        "pc_cols": pc_cols,
        "outcome": outcome,
        "sector_cols": sector_cols,
        "apply_sqrt_transform": apply_sqrt_transform,
        "sqrt_offset": sqrt_offset,
        # POPULATION scaling info for CV / forecasting
        "pop_col": pop_col,
        "pop_mean": float(pop_mean),
        "pop_std": float(pop_std),
        "has_population": True,
    }

    return series_dict, meta



def fit_sarimax_single(
    y: pd.Series,
    exog: pd.DataFrame,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 12),
):
    """
    Fit a SARIMAX model for a single department on the (possibly transformed) y.
    """

    model = sm.tsa.SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return result


def fit_sarimax_panel(
    series_dict,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 12),
):
    """
    Fit SARIMAX for all departments in the panel.

    series_dict[dept]["y"] is assumed to be already transformed (e.g. sqrt).
    """

    results_dict = {}
    rows = []

    for dept, data in series_dict.items():
        print(f"Fitting SARIMAX for dept {dept}...")
        y = data["y"]
        exog = data["exog"]

        res = fit_sarimax_single(y, exog, order=order, seasonal_order=seasonal_order)
        results_dict[dept] = res

        rows.append(
            {
                "DEPT_CODE": dept,
                "aic": res.aic,
                "bic": res.bic,
                "llf": res.llf,
                "nobs": res.nobs,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("DEPT_CODE").reset_index(drop=True)
    return results_dict, summary_df


def sarimax_diagnostics(result, lags=24, title_prefix=""):
    """
    Diagnostics for a single SARIMAX result on the transformed scale:
    - residuals over time
    - ACF & PACF of residuals
    - QQ plot
    - Ljung–Box p-values
    """
    resid = result.resid.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    # 1. Residuals over time
    resid.plot(ax=axes[0])
    axes[0].set_title(f"{title_prefix} Residuals over time")
    axes[0].axhline(0, color="black", linewidth=1)

    # 2. ACF
    plot_acf(resid, ax=axes[1], lags=lags)
    axes[1].set_title(f"{title_prefix} Residual ACF")

    # 3. PACF
    plot_pacf(resid, ax=axes[2], lags=lags, method="ywm")
    axes[2].set_title(f"{title_prefix} Residual PACF")

    # 4. QQ plot (should look *much* closer to straight line now)
    sm.qqplot(resid, line="45", ax=axes[3])
    axes[3].set_title(f"{title_prefix} Residual QQ plot")

    plt.tight_layout()
    plt.show()

    # Ljung–Box test for residual autocorrelation
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
    print(f"{title_prefix} Ljung-Box test (lag {lags}):")
    print(lb)


def panel_aic_bic(summary_df: pd.DataFrame, k_per_model: int | None = None):
    """
    Compute panel-level AIC/BIC by summing log-likelihoods across departments.
    (Still valid; likelihood is for the transformed model.)
    """
    llf_total = summary_df["llf"].sum()
    nobs_total = summary_df["nobs"].sum()

    if k_per_model is None:
        k_est = (summary_df["aic"] + 2 * summary_df["llf"]) / 2
        k_total = k_est.sum()
    else:
        k_total = k_per_model * len(summary_df)

    AIC_total = 2 * k_total - 2 * llf_total
    BIC_total = k_total * np.log(nobs_total) - 2 * llf_total

    return float(llf_total), float(AIC_total), float(BIC_total)


def build_sarimax_residuals_geodf(
    results_dict,
    geojson_path: str,
    dept_code_col_geo: str = "dept_code_hecho",
):
    """
    Build a GeoDataFrame with sum (and mean) of absolute SARIMAX residuals
    per department.

    Parameters
    ----------
    results_dict : dict[int, SARIMAXResults]
        Output of fit_sarimax_panel; keys are DEPT_CODE.
    geojson_path : str
        Path to 'colombia_departments.geojson'.
    dept_code_col_geo : str
        Column name in the geojson with dept codes (e.g. 'dept_code_hecho').

    Returns
    -------
    gdf_resid : geopandas.GeoDataFrame
        Columns: DEPT_CODE, sum_abs_resid, mean_abs_resid, geometry, dept_name_hecho
    """

    # 1) Aggregate residuals per department
    rows = []
    for dept, res in results_dict.items():
        resid = res.resid.dropna().values.astype(float)
        sum_abs = np.abs(resid).sum()
        mean_abs = np.abs(resid).mean()
        rows.append(
            {
                "DEPT_CODE": int(dept),
                "sum_abs_resid": sum_abs,
                "mean_abs_resid": mean_abs,
            }
        )
    df_resid = pd.DataFrame(rows)

    # 2) Read geojson with department shapes
    gdf_dept = gpd.read_file(geojson_path)
    gdf_dept[dept_code_col_geo] = gdf_dept[dept_code_col_geo].astype(int)

    # 3) Merge residual summary with geometries
    gdf_resid = gdf_dept.merge(
        df_resid,
        left_on=dept_code_col_geo,
        right_on="DEPT_CODE",
        how="left",
    )

    gdf_resid = gpd.GeoDataFrame(gdf_resid, geometry="geometry", crs="EPSG:4326")

    return gdf_resid


def plot_sarimax_spatial_residuals(
    gdf_resid: gpd.GeoDataFrame,
    value_col: str = "sum_abs_resid",
    title: str = "SARIMAX: Sum of absolute residuals by department",
):
    """
    Plot a static choropleth of aggregated SARIMAX residuals.

    Parameters
    ----------
    gdf_resid : GeoDataFrame
        Output of build_sarimax_residuals_geodf.
    value_col : str
        Column to color by (default: 'sum_abs_resid').
    title : str
        Plot title.
    """

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


def build_pc_sign_geodf(
    results_dict,
    meta,
    geojson_path: str,
    alpha: float = 0.05,
    dept_code_col_geo: str = "dept_code_hecho",
):
    """
    Build a GeoDataFrame with sign-coded PC coefficients per department:
      +1 if beta > 0 & p <= alpha
      -1 if beta < 0 & p <= alpha
      0  otherwise (including non-significant or missing)

    Parameters
    ----------
    results_dict : dict[int, SARIMAXResults]
        Output of fit_sarimax_panel: dept_code -> SARIMAXResults
    meta : dict
        Output of prepare_panel_for_sarimax; must include "pc_cols", "dept_codes"
    geojson_path : str
        Path to colombia_departments.geojson
    alpha : float
        Significance level for p-values (default 0.05)
    dept_code_col_geo : str
        Column name in geojson for department code (string like "05")

    Returns
    -------
    gdf_pc_sign : GeoDataFrame
        Columns: DEPT_CODE, one column per PC with values {-1, 0, 1}, geometry, dept_name_hecho
    """

    pc_cols = meta["pc_cols"]        # e.g. ["PC1", "PC2", "PC3"]
    dept_codes = meta["dept_codes"]  # array of department codes

    rows = []

    for dept in dept_codes:
        res = results_dict.get(dept)
        if res is None:
            # no result for this dept; set all zeros
            sign_row = {pc: 0 for pc in pc_cols}
        else:
            params = res.params
            pvals = res.pvalues

            sign_row = {}
            for pc in pc_cols:
                if pc in params.index:
                    beta = params[pc]
                    pval = pvals[pc]
                    if pval <= alpha:
                        sign_row[pc] = 1 if beta > 0 else -1
                    else:
                        sign_row[pc] = 0
                else:
                    # PC not in this model for some reason
                    sign_row[pc] = 0

        sign_row["DEPT_CODE"] = dept
        rows.append(sign_row)

    df_sign = pd.DataFrame(rows)

    # Read geojson and merge geometries
    gdf_dept = gpd.read_file(geojson_path)
    gdf_dept[dept_code_col_geo] = gdf_dept[dept_code_col_geo].astype(int)

    gdf_pc_sign = df_sign.merge(
        gdf_dept[[dept_code_col_geo, "dept_name_hecho", "geometry"]],
        left_on="DEPT_CODE",
        right_on=dept_code_col_geo,
        how="left",
    )

    gdf_pc_sign = gpd.GeoDataFrame(gdf_pc_sign, geometry="geometry", crs="EPSG:4326")
    return gdf_pc_sign


def plot_pc_sign_maps(
    gdf_pc_sign,
    pc_cols=None,
    title_prefix="PC sign map",
):
    """
    Plot one spatial map per PC, where color in {-1, 0, 1} indicates:
      -1: significant negative coefficient (p <= 0.05)
       0: not significant
       1: significant positive coefficient

    Parameters
    ----------
    gdf_pc_sign : GeoDataFrame
        Output of build_pc_sign_geodf.
    pc_cols : list[str] or None
        Which PC columns to plot. If None, uses all starting with "PC".
    title_prefix : str
        Title prefix for each map.
    """

    if pc_cols is None:
        pc_cols = [c for c in gdf_pc_sign.columns if c.startswith("PC")]

    for pc in pc_cols:
        fig = px.choropleth(
            gdf_pc_sign,
            geojson=gdf_pc_sign.geometry,
            locations=gdf_pc_sign.index,
            color=pc,
            color_continuous_scale=["blue", "lightgray", "red"],  # -1,0,1
            range_color=(-1, 1),
            hover_name="dept_name_hecho",
            scope="south america",
            title=f"{title_prefix}: {pc}",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.show()




def extract_significance_map(results_dict, param_name: str):
    """
    For a given parameter name (e.g. "PC1", "PC2", "ar.L1", "ar.S.L12"),
    extract a sign/significance map over departments:

      +1  if beta > 0 and p <= 0.05
      -1  if beta < 0 and p <= 0.05
       0  if p > 0.05 or parameter not present

    Parameters
    ----------
    results_dict : dict[int, SARIMAXResults]
        Output from fit_sarimax_panel (dept_code -> results object).
    param_name : str
        Exact name of the parameter in res.params.index.

    Returns
    -------
    df_sign : pd.DataFrame
        Columns: DEPT_CODE, sign_value
    """
    rows = []

    for dept, res in results_dict.items():
        params = res.params
        pvals = res.pvalues

        if param_name not in params.index:
            # parameter not in this department's model (e.g., dropped)
            sign_value = 0
        else:
            beta = params[param_name]
            pval = pvals[param_name]

            if pval > 0.05:
                sign_value = 0
            elif beta > 0:
                sign_value = 1
            elif beta < 0:
                sign_value = -1
            else:
                sign_value = 0

        rows.append({"DEPT_CODE": dept, "sign_value": sign_value})

    df_sign = pd.DataFrame(rows)
    return df_sign


def plot_sign_map(
    df_sign: pd.DataFrame,
    gdf_dept: gpd.GeoDataFrame,
    title: str = "",
    dept_code_col_geo: str = "dept_code_hecho",
):
    """
    Merge the sign DataFrame with the department GeoDataFrame
    and plot a choropleth of sign_value (-1, 0, 1).

    Parameters
    ----------
    df_sign : DataFrame
        Columns: DEPT_CODE, sign_value
    gdf_dept : GeoDataFrame
        Columns: dept_code_hecho (or similar), dept_name_hecho, geometry
    title : str
    dept_code_col_geo : str
        Name of department code column in gdf_dept.
    """

    # Ensure numeric dept codes align
    gdf = gdf_dept.copy()
    gdf[dept_code_col_geo] = gdf[dept_code_col_geo].astype(int)

    gdf_merged = gdf.merge(
        df_sign,
        left_on=dept_code_col_geo,
        right_on="DEPT_CODE",
        how="left",
    )

    # default 0 if missing
    gdf_merged["sign_value"] = gdf_merged["sign_value"].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    gdf_merged.plot(
        column="sign_value",
        ax=ax,
        legend=True,
        cmap="bwr",      # blue = negative, white = 0, red = positive
        vmin=-1,
        vmax=1,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_axis_off()
    ax.set_title(title)
    plt.show()


def plot_pca_betas_spatially(results_dict, gdf_dept, pc_cols):
    """
    Plot one map per principal component, showing:

      +1  if PC coefficient is significantly positive (p <= 0.05)
      -1  if significantly negative
       0  if not significant or not present

    Parameters
    ----------
    results_dict : dict[int, SARIMAXResults]
        From fit_sarimax_panel.
    gdf_dept : GeoDataFrame
        Department geometries.
    pc_cols : list[str]
        Names of PC regressors, e.g. ["PC1", "PC2", "PC3"].
    """
    for pc in pc_cols:
        df_sign = extract_significance_map(results_dict, param_name=pc)
        plot_sign_map(df_sign, gdf_dept, title=f"Significant sign of {pc} coefficient")


def plot_seasonal_coeffs_spatially(results_dict, gdf_dept, season_names):
    for s in season_names:
        df_sign = extract_significance_map(results_dict, param_name=s)
        plot_sign_map(df_sign, gdf_dept, title=f"Significant sign of {s} (seasonal) coefficient")


def plot_ar_coeffs_spatially(results_dict, gdf_dept, ar_lags=[1]):
    for lag in ar_lags:
        name = f"ar.L{lag}"
        df_sign = extract_significance_map(results_dict, name)
        plot_sign_map(df_sign, gdf_dept, f"AR({lag}) coefficient sign")

def plot_ma_coeffs_spatially(results_dict, gdf_dept, ma_lags=[1]):
    for lag in ma_lags:
        name = f"ma.L{lag}"
        df_sign = extract_significance_map(results_dict, name)
        plot_sign_map(df_sign, gdf_dept, f"MA({lag}) coefficient sign")


def sarimax_in_sample_cv_panel(
    series_dict,
    results_dict,
    start_frac: float = 0.5,
):
    """
    In-sample one-step-ahead cross-validation for SARIMAX panel.

    For each department:
      - uses fitted SARIMAXResults
      - computes one-step-ahead predictions for the last (1 - start_frac) fraction
        of the sample (dynamic=False)
      - returns RMSE, MAE, NLL per dept and overall.

    Parameters
    ----------
    series_dict : dict[int, dict]
        Output of prepare_panel_for_sarimax (dept -> {"y","exog"}).
    results_dict : dict[int, SARIMAXResults]
        Output of fit_sarimax_panel.
    start_frac : float
        Fraction of the time series to skip before evaluating predictions
        (e.g. 0.5 means start evaluating on the second half).

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per dept with rmse, mae, poisson_nll.
    """

    rows = []

    for dept, data in series_dict.items():
        if dept not in results_dict:
            continue

        res = results_dict[dept]
        y = data["y"]
        exog = data["exog"]

        n = len(y)
        start_idx = int(n * start_frac)
        start_date = y.index[start_idx]

        pred = res.get_prediction(
            start=start_date,
            end=y.index[-1],
            exog=exog.iloc[start_idx:],
            dynamic=False,  # full one-step-ahead
        )

        y_true = y.iloc[start_idx:].astype(float)
        y_pred = pred.predicted_mean.astype(float)

        eps = 1e-8
        y_pred = np.clip(y_pred, eps, None)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        poisson_nll = np.mean(y_pred - y_true * np.log(y_pred))

        rows.append(
            {
                "DEPT_CODE": dept,
                "rmse": rmse,
                "mae": mae,
                "poisson_nll": poisson_nll,
                "n_eval": len(y_true),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("DEPT_CODE").reset_index(drop=True)
    return metrics_df



def sarimax_rolling_cv_panel(
    df_polars: pl.DataFrame,
    outcome: str,
    sector_cols=None,
    n_pcs: int = 3,
    order=(2, 0, 2),
    seasonal_order=(1, 0, 1, 12),
    n_folds: int = 3,
    horizon_months: int = 6,
):
    """
    Rolling-origin time-series CV for SARIMAX with PCA exogenous variables
    and standardized POPULATION.
    """

    if sector_cols is None:
        sector_cols = [f"S0{i}" for i in range(1, 9)]

    # sort by time
    df_polars = df_polars.sort(["YYYYMM", "DEPT_CODE"])
    time_vals = np.sort(df_polars.select("YYYYMM").to_series().to_list())
    n_time = len(time_vals)

    fold_cut_indices = np.linspace(
        int(n_time * 0.4),
        n_time - horizon_months - 1,
        n_folds,
        dtype=int,
    )

    rows = []

    for fold, cut_idx in enumerate(fold_cut_indices, start=1):
        cutoff_time = time_vals[cut_idx]
        print(f"\n=== SARIMAX CV fold {fold}/{n_folds}: train <= {cutoff_time} ===")

        # TRAIN set: up to cutoff
        df_train = df_polars.filter(pl.col("YYYYMM") <= cutoff_time)
        series_train, meta_train = prepare_panel_for_sarimax(
            df_train, outcome=outcome, sector_cols=sector_cols, n_pcs=n_pcs
        )
        results_train, _ = fit_sarimax_panel(
            series_train, order=order, seasonal_order=seasonal_order
        )

        # TEST set: next horizon_months after cutoff
        future_times = time_vals[cut_idx + 1 : cut_idx + 1 + horizon_months]
        df_test = df_polars.filter(pl.col("YYYYMM").is_in(future_times))
        df_test_pd = df_test.to_pandas().reset_index(drop=True)

        # Transform test sector variables with training scaler + PCA
        scaler = meta_train["scaler"]
        pca = meta_train["pca"]
        pc_cols = meta_train["pc_cols"]

        X_sectors_test = df_test_pd[sector_cols].values.astype(float)
        X_scaled_test = scaler.transform(X_sectors_test)
        X_pcs_test = pca.transform(X_scaled_test)

        exog_test = pd.DataFrame(X_pcs_test, columns=pc_cols, index=df_test_pd.index)

        # Add standardized POPULATION using training mean/std
        if meta_train.get("has_population", False):
            pop_mean = meta_train["pop_mean"]
            pop_std = meta_train["pop_std"]
            pop_col = meta_train["pop_col"]

            pop_raw_test = df_test_pd["POPULATION"].astype(float).values
            pop_stdzd_test = (pop_raw_test - pop_mean) / pop_std
            exog_test[pop_col] = pop_stdzd_test

        # Collect predictions across departments
        y_true_all = []
        y_pred_all = []

        for dept in meta_train["dept_codes"]:
            res = results_train[dept]

            mask_dept = df_test_pd["DEPT_CODE"] == dept
            df_test_d = df_test_pd[mask_dept].copy()
            if df_test_d.empty:
                continue

            y_true = df_test_d[outcome].astype(float)
            exog_d = exog_test.loc[df_test_d.index]

            # Forecast for this dept
            res_d = res.get_forecast(steps=len(y_true), exog=exog_d)
            y_pred = res_d.predicted_mean.astype(float)

            y_true_all.append(y_true.values)
            y_pred_all.append(y_pred.values)

        if not y_true_all:
            continue

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        eps = 1e-8
        y_pred_all = np.clip(y_pred_all, eps, None)

        rmse = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
        mae = np.mean(np.abs(y_true_all - y_pred_all))
        poisson_nll = np.mean(y_pred_all - y_true_all * np.log(y_pred_all))

        rows.append(
            dict(
                model="SARIMAX",
                outcome=outcome,
                fold=fold,
                cutoff_time=cutoff_time,
                horizon_months=horizon_months,
                rmse=rmse,
                mae=mae,
                poisson_nll=poisson_nll,
            )
        )

        print(
            f"Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, Poisson-NLL={poisson_nll:.3f}"
        )

    cv_results = pd.DataFrame(rows)
    return cv_results



def sarimax_gof(result, lags: int = 24, alpha: float = 0.05, title_prefix: str = ""):
    """
    Goodness-of-fit tests for a single SARIMAX model.

    Tests
    -----
    1) Ljung–Box test for residual autocorrelation up to `lags`
    2) Jarque–Bera test for normality of residuals

    Parameters
    ----------
    result : SARIMAXResults
    lags : int
        Max lag for Ljung–Box.
    alpha : float
        Significance level for interpretation.
    title_prefix : str
        Optional label (e.g. department code).

    Returns
    -------
    gof : dict
        {
            "lb_stat": float,
            "lb_pvalue": float,
            "jb_stat": float,
            "jb_pvalue": float
        }
    """

    resid = result.resid.dropna()

    # Ljung–Box (use all lags up to `lags`, look at the last one)
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])

    # Jarque–Bera normality test
    jb_stat, jb_pvalue, _, _ = jarque_bera(resid)

    print(f"{title_prefix} Ljung–Box (lag {lags}): stat={lb_stat:.3f}, p={lb_pvalue:.3f}")
    if lb_pvalue < alpha:
        print(f"  -> Reject H0: residuals show autocorrelation (bad).")
    else:
        print(f"  -> Fail to reject H0: no strong evidence of residual autocorrelation.")

    print(f"{title_prefix} Jarque–Bera: stat={jb_stat:.3f}, p={jb_pvalue:.3f}")
    if jb_pvalue < alpha:
        print(f"  -> Reject H0: residuals deviate from normality (heavy tails / skew).")
    else:
        print(f"  -> Fail to reject H0: residuals are roughly normal.")

    gof = {
        "lb_stat": lb_stat,
        "lb_pvalue": lb_pvalue,
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
    }
    return gof
