# spacetime_car.py

import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from sklearn.decomposition import PCA
from scipy.stats import norm 

# Optional spatial diagnostics
try:
    import libpysal
    from esda.moran import Moran
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False


# -------------------------------------------------------------------
# 1. DATA PREPARATION
# -------------------------------------------------------------------


def prepare_panel_for_st_car(
    df_polars: pl.DataFrame,
    outcome: str,
    x_sector_cols=None,
    use_pca: bool = True,
    pca_var: float = 0.95,
    max_pcs: int | None = None,
    k_neighbors: int = 4,
):
    """
    Convert Polars panel to pandas + design matrix for CAR model.
    Optionally apply PCA to highly correlated sector columns (S0X),
    and add POPULATION as an extra standardized covariate (not in PCA).

    Returns
    -------
    df_pd : pandas.DataFrame
    y : np.ndarray (N_obs,)
    X_stdzd : np.ndarray (N_obs, K)
    meta : dict
    """

    if x_sector_cols is None:
        x_sector_cols = [f"S0{i}" for i in range(1, 9)]

    # basic cleaning and sorting
    df_polars = (
        df_polars
        .with_columns([
            pl.col("DEPT_CODE").cast(pl.Int32),
            pl.col("YYYYMM").cast(pl.Date),
        ])
        .sort(["DEPT_CODE", "YYYYMM"])
    )

    df_pd = df_polars.to_pandas().reset_index(drop=True)

    # indices
    dept_codes = np.sort(df_pd["DEPT_CODE"].unique())
    time_vals = np.sort(df_pd["YYYYMM"].unique())
    n_dept = len(dept_codes)
    n_time = len(time_vals)

    dept_to_idx = {d: i for i, d in enumerate(dept_codes)}
    time_to_idx = {t: i for i, t in enumerate(time_vals)}

    df_pd["dept_idx"] = df_pd["DEPT_CODE"].map(dept_to_idx)
    df_pd["time_idx"] = df_pd["YYYYMM"].map(time_to_idx)
    df_pd["month_idx"] = df_pd["YYYYMM"].dt.month - 1

    # outcome
    y = df_pd[outcome].astype("int64").values

    # ---------- PCA on sector columns (S0X only) ----------
    X_sector = df_pd[x_sector_cols].to_numpy().astype(float)

    # standardize sectors before PCA
    sector_mean = X_sector.mean(axis=0, keepdims=True)
    sector_std = X_sector.std(axis=0, keepdims=True)
    sector_std[sector_std == 0] = 1.0
    X_sector_std = (X_sector - sector_mean) / sector_std

    if use_pca:
        pca = PCA() if max_pcs is None else PCA(n_components=max_pcs)
        pca.fit(X_sector_std)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        k = np.searchsorted(cum_var, pca_var) + 1
        if max_pcs is not None:
            k = min(k, max_pcs)

        components = pca.components_[:k]            # (k, n_sector)
        X_pcs = X_sector_std @ components.T        # (N_obs, k)

        X_design = X_pcs
        x_cols_effective = [f"PC{i+1}" for i in range(k)]

        pca_info = {
            "enabled": True,
            "sector_cols": x_sector_cols,
            "sector_mean": sector_mean,
            "sector_std": sector_std,
            "components": components,
            "n_components": k,
            "explained_variance_ratio": pca.explained_variance_ratio_[:k],
        }
    else:
        # no PCA: use raw sector columns directly
        X_design = X_sector
        x_cols_effective = x_sector_cols
        pca_info = {"enabled": False}

    # ---------- add POPULATION as extra covariate (not in PCA) ----------
    if "POPULATION" not in df_pd.columns:
        raise KeyError("Column 'POPULATION' not found in df_polars / df_pd.")

    pop = df_pd["POPULATION"].to_numpy().astype(float).reshape(-1, 1)

    # Combine design matrix: [sectors/PCs , POPULATION]
    X_full = np.hstack([X_design, pop])
    x_cols_effective = x_cols_effective + ["POPULATION"]

    # ---------- standardize final design matrix ----------
    X_mean = X_full.mean(axis=0, keepdims=True)
    X_std = X_full.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_stdzd = (X_full - X_mean) / X_std

    N, K = X_stdzd.shape

    # build spatial adjacency from coordinates
    coords = (
        df_pd
        .drop_duplicates(subset=["DEPT_CODE"])
        .set_index("DEPT_CODE")[["DEPT_LAT", "DEPT_LON"]]
        .loc[dept_codes]
        .to_numpy()
    )

    W_sparse = kneighbors_graph(coords, n_neighbors=k_neighbors,
                                mode="connectivity", include_self=False)
    W_raw = W_sparse.toarray().astype(float)
    W_raw = np.maximum(W_raw, W_raw.T)
    D = np.diag(W_raw.sum(axis=1))

    meta = {
        "dept_codes": dept_codes,
        "time_vals": time_vals,
        "dept_to_idx": dept_to_idx,
        "time_to_idx": time_to_idx,
        "coords": coords,
        "X_cols": x_cols_effective,   # PC names (or sectors) + POPULATION
        "X_mean": X_mean,
        "X_std": X_std,
        "W_raw": W_raw,
        "D": D,
        "n_dept": n_dept,
        "n_time": n_time,
        "N": N,
        "K": K,
        "outcome": outcome,
        "pca": pca_info,
        "has_population": True,
        "population_col": "POPULATION",
    }

    return df_pd, y, X_stdzd, meta




# -------------------------------------------------------------------
# 2.1 BUILD SPACE–TIME CAR POISSON MODEL
# -------------------------------------------------------------------

def build_st_car_poisson_model(df_pd, y, X_stdzd, meta, epsilon=1e-5):
    """
    Build a PyMC space–time CAR Poisson model:
        log(mu_{i,t}) = alpha + X_{i,t} beta + u_i + v_t

    Parameters
    ----------
    df_pd : pandas.DataFrame
        Panel data with dept_idx and time_idx columns.
    y : np.ndarray
        Outcome vector (N,).
    X_stdzd : np.ndarray
        Standardized covariates (N, K).
    meta : dict
        Metadata from prepare_panel_for_st_car.
    epsilon : float
        Small diagonal jitter for numerical stability in CAR precision.

    Returns
    -------
    model : pm.Model
        PyMC model object.
    """

    dept_idx = df_pd["dept_idx"].values
    time_idx = df_pd["time_idx"].values

    n_dept = meta["n_dept"]
    n_time = meta["n_time"]
    K = meta["K"]
    D = meta["D"]
    W_raw = meta["W_raw"]

    coords_model = {
        "obs": np.arange(len(y)),
        "dept": meta["dept_codes"],
        "time": meta["time_vals"],
        "cov": np.arange(K),
    }

    with pm.Model(coords=coords_model) as model:
        # Data containers
        X_data = pm.Data("X", X_stdzd, dims=("obs", "cov"))
        y_obs = pm.Data("y", y, dims="obs")
        dept_idx_data = pm.Data("dept_idx", dept_idx, dims="obs")
        time_idx_data = pm.Data("time_idx", time_idx, dims="obs")

        # Regression coefficients
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=2.0, dims="cov")

        # Spatial CAR effect: u
        rho_s = pm.Uniform("rho_s", lower=0.0, upper=0.99)
        tau_s = pm.HalfNormal("tau_s", sigma=1.0)

        D_pt = pt.as_tensor_variable(D)
        W_raw_pt = pt.as_tensor_variable(W_raw)

        Q = tau_s * (D_pt - rho_s * W_raw_pt + epsilon * pt.eye(n_dept))
        Sigma_u = pt.nlinalg.matrix_inverse(Q)

        u = pm.MvNormal(
            "u",
            mu=pt.zeros(n_dept),
            cov=Sigma_u,
            shape=n_dept,
            dims="dept",
        )

        # Temporal random walk: v_t
        sigma_t = pm.HalfNormal("sigma_t", sigma=1.0)
        v_raw = pm.Normal("v_raw", mu=0.0, sigma=1.0, dims="time")
        v = pm.Deterministic("v", sigma_t * pt.cumsum(v_raw), dims="time")

        # Linear predictor
        eta = (
            alpha
            + pt.sum(X_data * beta, axis=1)
            + u[dept_idx_data]
            + v[time_idx_data]
        )
        mu = pt.exp(eta)

        # Likelihood
        y_like = pm.Poisson("y_like", mu=mu, observed=y_obs, dims="obs")

    return model

# -------------------------------------------------------------------
# 2.2 BUILD SPACE–TIME CAR NEGATIVE BINOMIAL MODEL
# -------------------------------------------------------------------

def build_st_car_nb_model_icar(df_pd, y, X_stdzd, meta, epsilon=1e-8):
    """
    Space–time ICAR Negative Binomial model:

        log(mu_{i,t}) = alpha + X_{i,t} beta + u_i + v_t
        u ~ ICAR(τ) via neighbor differences
        v_t: temporal random walk
        y_{i,t} ~ NB(mu_{i,t}, alpha_nb)

    Uses a sparse ICAR prior via a Potential, no dense covariance.

    Parameters
    ----------
    df_pd : pandas.DataFrame
        Must contain 'dept_idx' and 'time_idx' columns.
    y : np.ndarray (N_obs,)
        Counts (flattened panel).
    X_stdzd : np.ndarray (N_obs, K)
        Standardized covariates for each observation.
    meta : dict
        From prepare_panel_for_st_car, must contain:
            - 'n_dept', 'n_time'
            - 'W_raw' (adjacency matrix)
            - 'dept_codes', 'time_vals'
    epsilon : float
        Small jitter to avoid degeneracies in RW.

    Returns
    -------
    model : pm.Model
    """

    dept_idx = df_pd["dept_idx"].values
    time_idx = df_pd["time_idx"].values
    month_idx = df_pd["month_idx"].values

    n_dept = meta["n_dept"]
    n_time = meta["n_time"]
    n_month = meta.get("n_month", 12)
    K = meta["K"]
    W_raw = meta["W_raw"]

    # Build edge list and weights from W_raw (sparse adjacency)
    # Use only i < j to avoid double counting (undirected graph)
    edges = np.array(np.where(W_raw > 0)).T  # all (i,j) where W>0
    edges = edges[edges[:, 0] < edges[:, 1]]
    w_ij = W_raw[edges[:, 0], edges[:, 1]]

    # To tensors
    edges_t = pt.as_tensor_variable(edges.astype("int64"))  # shape (E, 2)
    w_ij_t = pt.as_tensor_variable(w_ij.astype("float64"))  # (E,)

    coords_model = {
        "obs": np.arange(len(y)),
        "dept": meta["dept_codes"],
        "time": meta["time_vals"],
        "cov": np.arange(K),
        "month": np.arange(n_month),
    }

    with pm.Model(coords=coords_model) as model:
        # Data containers
        X_data = pm.Data("X", X_stdzd, dims=("obs", "cov"))
        y_obs = pm.Data("y", y, dims="obs")
        dept_idx_data = pm.Data("dept_idx", dept_idx, dims="obs")
        time_idx_data = pm.Data("time_idx", time_idx, dims="obs")
        month_idx_data = pm.Data("month_idx", month_idx, dims="obs")

        # Regression coefficients
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=2.0, dims="cov")

        # ---------------- SPATIAL ICAR PRIOR ----------------
        # Unconstrained spatial effect
        u_raw = pm.Normal("u_raw", mu=0.0, sigma=1.0, shape=n_dept, dims="dept")
        # Center to enforce identifiability (sum(u)=0)
        u = pm.Deterministic("u", u_raw - u_raw.mean(), dims="dept")

        # Precision (strength of spatial smoothing)
        tau_s = pm.HalfNormal("tau_s", sigma=1.0)

        # Differences across edges
        ui = u[edges_t[:, 0]]
        uj = u[edges_t[:, 1]]
        diff = ui - uj

        # ICAR potential: -0.5 * tau * sum w_ij * (u_i - u_j)^2
        pm.Potential("icar_spatial",
                     -0.5 * tau_s * pt.sum(w_ij_t * diff**2))

        # ---------------- TEMPORAL RANDOM WALK ----------------
        sigma_t = pm.Exponential("sigma_t", 5.0)  # stronger regularization

        v_raw = pm.Normal("v_raw", mu=0.0, sigma=1.0, dims="time")
        v = pm.Deterministic("v",
                             sigma_t * pt.cumsum(v_raw) - sigma_t * pt.cumsum(v_raw).mean(),
                             dims="time")

        # ---------------- SEASONAL MONTH EFFECT ----------------
        # unconstrained monthly effects, then sum-to-zero identifiability
        gamma_raw = pm.Normal("gamma_raw", mu=0.0, sigma=1.0, dims="month")
        gamma = pm.Deterministic(
            "gamma",
            gamma_raw - gamma_raw.mean(),
            dims="month",
        )

        # ---------------- LINEAR PREDICTOR ----------------
        eta = (
            alpha
            + pt.sum(X_data * beta, axis=1)
            + u[dept_idx_data]
            + v[time_idx_data]       # long-term trend
            + gamma[month_idx_data]  # seasonal effect
        )
        mu = pt.exp(eta)

        # ---------------- NEGATIVE BINOMIAL LIKELIHOOD ----------------
        alpha_nb = pm.HalfNormal("alpha_nb", sigma=2.0)

        y_like = pm.NegativeBinomial(
            "y_like",
            mu=mu,
            alpha=alpha_nb,
            observed=y_obs,
            dims="obs",
        )

    return model



# -------------------------------------------------------------------
# 3. FIT MODEL
# -------------------------------------------------------------------

def fit_st_car_model(
    model,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    max_treedepth: int = 15,
):

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            return_inferencedata=True,   # <-- CRUCIAL
            idata_kwargs={"log_likelihood": True},
        )

    return idata, idata



# -------------------------------------------------------------------
# 4. MODEL DIAGNOSTICS
# -------------------------------------------------------------------


def diagnostics_mcmc(idata, var_names=None):
    """
    Print basic MCMC diagnostics and show trace / energy plots.
    Safely ignores variable names that are not in the posterior.
    """
    if var_names is None:
        var_names = ["alpha", "beta", "tau_s", "sigma_t", "alpha_nb"]

    available = list(idata.posterior.data_vars.keys())
    used_var_names = [v for v in var_names if v in available]

    if not used_var_names:
        print("No requested variables found in idata.posterior; available:", available)
        return

    print(az.summary(idata, var_names=used_var_names, round_to=2))

    az.plot_trace(idata, var_names=used_var_names, compact=True)
    plt.show()

    az.plot_energy(idata)
    plt.show()


def diagnostics_ppc_residuals(model, idata, df_pd, y, var_name: str = "y_like"):
    """
    Posterior predictive checks + residual diagnostics for NB-CAR:
    - Observed vs predicted
    - Residual vs fitted (Pearson, using predictive variance)
    - QQ plot of Dunn–Smyth (randomized quantile) residuals
    - Predictive coverage
    """

    # 1) Posterior predictive sampling
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[var_name],
            return_inferencedata=True,
        )

    # 2) Extract PPC draws
    # shape: (chain, draw, obs)
    y_ppc = ppc.posterior_predictive[var_name].values
    # flatten chains + draws → (samples, obs)
    y_ppc_flat = y_ppc.reshape(-1, y_ppc.shape[-1])

    # 3) Posterior predictive mean & variance (for Pearson residuals)
    mu_hat = y_ppc_flat.mean(axis=0)
    var_hat = y_ppc_flat.var(axis=0, ddof=1)

    # Pearson residuals (using predictive variance)
    resid_pearson = (y - mu_hat) / np.sqrt(var_hat + 1e-8)

    # 4) Dunn–Smyth randomized quantile residuals
    # For each obs i, approximate F(y_i-1) and F(y_i) from PPC samples
    y_obs = np.asarray(y, dtype=int)
    S, N = y_ppc_flat.shape

    # preallocate
    dq_resid = np.zeros(N)

    for i in range(N):
        y_rep_i = y_ppc_flat[:, i]
        yi = y_obs[i]

        # empirical CDF at y_i - 1 and y_i
        cdf_lower = np.mean(y_rep_i < yi)        # F(y_i - 1)
        cdf_upper = np.mean(y_rep_i <= yi)       # F(y_i)

        # avoid degenerate 0 or 1
        cdf_lower = np.clip(cdf_lower, 1e-6, 1 - 1e-6)
        cdf_upper = np.clip(cdf_upper, 1e-6, 1 - 1e-6)

        # use midpoint (deterministic Dunn–Smyth); could also randomize
        u_i = 0.5 * (cdf_lower + cdf_upper)
        dq_resid[i] = norm.ppf(u_i)

    # 5) Observed vs predicted distribution
    plt.figure(figsize=(8, 4))
    plt.hist(y, bins=30, alpha=0.5, label="Observed")
    plt.hist(mu_hat, bins=30, alpha=0.5, label="Predicted mean")
    plt.legend()
    plt.title("Observed vs Predicted (posterior mean)")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.show()

    # 6) Observed vs predicted scatter (subset)
    idx_subset = np.random.choice(len(y), size=min(300, len(y)), replace=False)
    y_sub = y[idx_subset]
    mu_sub = mu_hat[idx_subset]

    plt.figure(figsize=(6, 6))
    plt.scatter(mu_sub, y_sub, alpha=0.5)
    min_val = min(mu_sub.min(), y_sub.min())
    max_val = max(mu_sub.max(), y_sub.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel("Predicted mean")
    plt.ylabel("Observed")
    plt.title("Observed vs Predicted (subset)")
    plt.show()

    # 7) Residuals vs fitted (Pearson)
    plt.figure(figsize=(7, 5))
    plt.scatter(mu_hat, resid_pearson, alpha=0.4)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Fitted mean (mu_hat)")
    plt.ylabel("Pearson residual (NB predictive var)")
    plt.title("Pearson residuals vs fitted (NB-CAR)")
    plt.show()

    # 8) QQ plot of Dunn–Smyth residuals
    sm.qqplot(dq_resid, line="45")
    plt.title("QQ plot of Dunn–Smyth residuals (NB-CAR)")
    plt.show()

    # 9) Predictive coverage
    lower = np.percentile(y_ppc_flat, 2.5, axis=0)
    upper = np.percentile(y_ppc_flat, 97.5, axis=0)
    covered = (y >= lower) & (y <= upper)
    coverage_rate = covered.mean()
    print(f"Empirical 95% predictive coverage: {coverage_rate:.3f}")

    # 10) AIC/BIC
    logL, AIC, BIC = compute_aic_bic(
        idata,
        param_names=[
            "alpha",
            "beta",
            "u_raw",
            "v_raw",
            "tau_s",
            "sigma_t",
            "alpha_nb",
            "gamma_raw",
        ],
    )
    print("logL:", logL)
    print("AIC:", AIC)
    print("BIC:", BIC)

    # return both residual types so you can use them elsewhere if needed
    return resid_pearson, mu_hat, dq_resid




def diagnostics_spatial_temporal(resid_pearson, df_pd, meta, max_lag=24):
    """
    Spatial and temporal diagnostics using residuals:
    - Moran's I on mean residual per dept
    - Temporal ACF of mean residual per time
    """
    dept_idx_arr = df_pd["dept_idx"].values
    time_idx_arr = df_pd["time_idx"].values
    n_dept = meta["n_dept"]
    n_time = meta["n_time"]
    coords = meta["coords"]

    # Mean residual per department
    resid_dept_mean = np.zeros(n_dept)
    for i in range(n_dept):
        mask = dept_idx_arr == i
        resid_dept_mean[i] = resid_pearson[mask].mean()

    if HAS_SPATIAL:
        w = libpysal.weights.KNN.from_array(coords, k=4)
        w.transform = "r"
        mi = Moran(resid_dept_mean, w)
        print(f"Moran's I (dept mean residual): {mi.I:.3f}, p-value: {mi.p_sim:.4f}")
    else:
        print("libpysal not installed, skipping Moran's I spatial diagnostic.")

    # Mean residual per time
    resid_time_mean = np.zeros(n_time)
    for t_idx in range(n_time):
        mask_t = time_idx_arr == t_idx
        resid_time_mean[t_idx] = resid_pearson[mask_t].mean()

    acf_vals_time = acf(resid_time_mean, fft=True, nlags=max_lag)

    plt.figure(figsize=(7, 4))
    plt.stem(range(len(acf_vals_time)), acf_vals_time, basefmt=" ")
    plt.xlabel("Lag (months)")
    plt.ylabel("ACF (mean residual)")
    plt.title("Temporal ACF of mean residuals")
    plt.show()


# -------------------------------------------------------------------
# 5. FORECASTING
# -------------------------------------------------------------------

def forecast_st_car(
    trace,
    df_pd,
    meta,
    horizon_months: int = 12,
    use_last_covariates: bool = True,
):
    """
    Forecast future counts for each department for a given horizon.
    Now includes POPULATION as an exogenous covariate (kept at last observed value).
    """

    dept_codes = meta["dept_codes"]
    time_vals = meta["time_vals"]
    n_dept = meta["n_dept"]
    n_time = meta["n_time"]
    K = meta["K"]
    X_cols = meta["X_cols"]
    X_mean = meta["X_mean"]
    X_std = meta["X_std"]

    pca_info = meta.get("pca", {"enabled": False})
    pca_enabled = pca_info.get("enabled", False)

    # Build future time axis
    last_time = time_vals[-1]
    future_times = [last_time + relativedelta(months=h) for h in range(1, horizon_months + 1)]

    # Build future panel (dept x future_times)
    rows = []
    for t in future_times:
        for d in dept_codes:
            rows.append({"DEPT_CODE": d, "YYYYMM": t})
    df_future = pd.DataFrame(rows)

    # Map indices
    df_future["dept_idx"] = df_future["DEPT_CODE"].map(meta["dept_to_idx"])
    time_idx_future_map = {t: n_time + i for i, t in enumerate(future_times)}
    df_future["time_idx"] = df_future["YYYYMM"].map(time_idx_future_map)

    # ---- fill covariates for future rows from last observed values ----
    if use_last_covariates:
        last_obs_by_dept = (
            df_pd.sort_values("YYYYMM")
            .groupby("DEPT_CODE")
            .tail(1)
            .set_index("DEPT_CODE")
        )

        if pca_enabled:
            sector_cols = pca_info["sector_cols"]
            for col in sector_cols:
                df_future[col] = df_future["DEPT_CODE"].map(last_obs_by_dept[col])
        else:
            # when not using PCA, X_cols already includes sectors + POPULATION
            for col in X_cols:
                if col in last_obs_by_dept.columns:
                    df_future[col] = df_future["DEPT_CODE"].map(last_obs_by_dept[col])

        # always propagate POPULATION explicitly (not part of PCA)
        if "POPULATION" in last_obs_by_dept.columns:
            df_future["POPULATION"] = df_future["DEPT_CODE"].map(
                last_obs_by_dept["POPULATION"]
            )
    else:
        raise NotImplementedError("Provide custom future covariates.")

    # --- build future design matrix [PCs/sectors , POPULATION] ---
    if pca_enabled:
        sector_cols = pca_info["sector_cols"]
        S_future = df_future[sector_cols].to_numpy().astype(float)

        sector_mean = pca_info["sector_mean"]
        sector_std = pca_info["sector_std"]
        components = pca_info["components"]    # (k, n_sector)

        S_future_std = (S_future - sector_mean) / sector_std
        X_pcs_future = S_future_std @ components.T  # (N_future, k)

        # append POPULATION as raw covariate in the same order as training
        pop_future = df_future["POPULATION"].to_numpy().astype(float).reshape(-1, 1)
        X_future_raw = np.hstack([X_pcs_future, pop_future])
    else:
        # X_cols already includes POPULATION as last column
        X_future_raw = df_future[X_cols].to_numpy().astype(float)

    # standardize with training X_mean, X_std
    X_future_std = (X_future_raw - X_mean) / X_std

    # Extract posterior draws
    alpha_post = trace.posterior["alpha"].values.reshape(-1)         # (S,)
    beta_post = trace.posterior["beta"].values.reshape(-1, K)        # (S,K)
    u_post = trace.posterior["u"].values.reshape(-1, n_dept)         # (S,n_dept)
    v_post = trace.posterior["v"].values.reshape(-1, n_time)         # (S,n_time)
    sigma_t_post = trace.posterior["sigma_t"].values.reshape(-1)     # (S,)

    n_samp = alpha_post.shape[0]
    H = horizon_months
    n_future_obs = df_future.shape[0]

    # Extend temporal random walk into future
    v_extended = np.zeros((n_samp, n_time + H))
    v_extended[:, :n_time] = v_post

    for s in range(n_samp):
        v_T = v_post[s, -1]
        sigma = sigma_t_post[s]
        increments = np.random.normal(0.0, sigma, size=H)
        v_future = v_T + np.cumsum(increments)
        v_extended[s, n_time:] = v_future

    dept_idx_future = df_future["dept_idx"].values
    time_idx_future = df_future["time_idx"].values

    y_future_samples = np.zeros((n_samp, n_future_obs))

    for s in range(n_samp):
        alpha_s = alpha_post[s]
        beta_s = beta_post[s]
        u_s = u_post[s]
        v_s_ext = v_extended[s]

        eta_future = (
            alpha_s
            + X_future_std @ beta_s
            + u_s[dept_idx_future]
            + v_s_ext[time_idx_future]
        )
        mu_future = np.exp(eta_future)

        # still using Poisson for predictive draws; if you want NB, swap here
        y_future_samples[s, :] = np.random.poisson(mu_future)

    # Summarize posterior predictive
    df_forecast = df_future.copy()
    df_forecast["pred_mean"] = y_future_samples.mean(axis=0)
    df_forecast["pred_lower_95"] = np.percentile(y_future_samples, 2.5, axis=0)
    df_forecast["pred_upper_95"] = np.percentile(y_future_samples, 97.5, axis=0)

    return df_forecast





# -------------------------------------------------------------------
# 6. CROSS VALIDATION
# -------------------------------------------------------------------

def car_in_sample_cv(
    model,
    idata,
    df_pd: pd.DataFrame,
    y: np.ndarray,
    var_name: str = "y_like",
):
    """
    In-sample cross-validation style evaluation for a CAR model.

    Uses posterior predictive distributions to compute:
        - RMSE
        - MAE
        - Poisson-style NLL
        - empirical 95% predictive coverage

    Parameters
    ----------
    model : pm.Model
        Fitted PyMC model (e.g. NB-CAR).
    idata : arviz.InferenceData
        Result of fit_st_car_model(..., return_inferencedata=True).
    df_pd : pandas.DataFrame
        Panel used for fitting (ordering must match y).
    y : np.ndarray
        Observed counts in the same order as df_pd / model data.
    var_name : str
        Name of the likelihood RV (default "y_like").

    Returns
    -------
    metrics : dict
        {
          "rmse": ...,
          "mae": ...,
          "poisson_nll": ...,
          "coverage_95": ...,
        }
    resid_pearson : np.ndarray
        Pearson residuals (N_obs,).
    mu_hat : np.ndarray
        Posterior predictive means (N_obs,).
    """

    # posterior predictive residuals + mean, var
    resid_pearson, mu_hat, var_hat = compute_residuals_pearson(
        model, idata, y, var_name=var_name
    )

    y_true = y.astype(float)
    y_pred = mu_hat.astype(float)
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, None)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    # simple Poisson-style NLL as a generic score
    poisson_nll = np.mean(y_pred - y_true * np.log(y_pred))

    # empirical 95% coverage from PPC
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[var_name],
            return_inferencedata=True,
            progressbar=False,
        )
    y_ppc = ppc.posterior_predictive[var_name].values
    y_ppc_flat = y_ppc.reshape(-1, y_ppc.shape[-1])
    lower = np.percentile(y_ppc_flat, 2.5, axis=0)
    upper = np.percentile(y_ppc_flat, 97.5, axis=0)
    covered = (y_true >= lower) & (y_true <= upper)
    coverage_95 = covered.mean()

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "poisson_nll": float(poisson_nll),
        "coverage_95": float(coverage_95),
    }
    return metrics, resid_pearson, mu_hat


def rolling_cv_car(
    df_polars: pl.DataFrame,
    outcome: str,
    model_builder,
    forecast_func,
    n_folds: int = 3,
    horizon_months: int = 6,
    draws: int = 1000,
    tune: int = 1000,
):
    """
    Rolling-origin time-series cross-validation for a CAR model.

    Parameters
    ----------
    df_polars : pl.DataFrame
        Full panel (all months).
    outcome : str
        Column to model (e.g. "NUMBER_OF_HOMICIDIO").
    model_builder : callable
        Function (df_pd, y, X_stdzd, meta) -> PyMC model.
        e.g. build_st_car_nb_model_icar.
    forecast_func : callable
        Function (idata, df_pd, meta, horizon_months) -> df_forecast
        (like forecast_st_car or forecast_st_car_zinb).
    n_folds : int
        Number of CV folds (cut points).
    horizon_months : int
        Forecast horizon for each fold.
    draws, tune : int
        MCMC settings for each fold (keep modest to save time).

    Returns
    -------
    cv_results : pandas.DataFrame
        One row per fold with RMSE/MAE/NLL.
    """

    # ensure sorted by time
    df_polars = df_polars.sort(["YYYYMM", "DEPT_CODE"])

    # unique months (as Polars Date)
    time_vals = np.sort(df_polars.select("YYYYMM").to_series().to_list())
    n_time = len(time_vals)

    # choose cut points (avoid too-early and too-late splits)
    fold_cut_indices = np.linspace(
        int(n_time * 0.4),
        n_time - horizon_months - 1,
        n_folds,
        dtype=int,
    )

    rows = []

    for fold, cut_idx in enumerate(fold_cut_indices, start=1):
        cutoff_time = time_vals[cut_idx]
        print(f"\n=== CV fold {fold}/{n_folds}: train <= {cutoff_time} ===")

        # TRAIN data: months <= cutoff_time
        df_train = df_polars.filter(pl.col("YYYYMM") <= cutoff_time)

        df_pd_train, y_train, X_train_stdzd, meta_train = prepare_panel_for_st_car(
            df_polars=df_train,
            outcome=outcome,
            x_sector_cols=[f"S0{i}" for i in range(1, 9)],  # S01..S08
            k_neighbors=4,
        )

        # build and fit model on training set
        model = model_builder(df_pd_train, y_train, X_train_stdzd, meta_train)
        idata, _ = fit_st_car_model(
            model,
            draws=draws,
            tune=tune,
            chains=2,          # modest for CV
            target_accept=0.9,
        )

        # FORECAST horizon_months ahead from cutoff
        df_forecast = forecast_func(idata, df_pd_train, meta_train, horizon_months=horizon_months)

        # ACTUAL data for comparison
        future_times = sorted(df_forecast["YYYYMM"].unique())
        # Cast to Polars Date so it matches df_polars["YYYYMM"]
        future_times_pl = pl.Series(future_times).cast(pl.Date)

        df_test = df_polars.filter(pl.col("YYYYMM").is_in(future_times_pl))

        # align by DEPT_CODE + YYYYMM
        df_test_pd = df_test.to_pandas()
        df_fore_pd = df_forecast[["DEPT_CODE", "YYYYMM", "pred_mean"]].copy()

        merged = df_test_pd.merge(
            df_fore_pd,
            left_on=["DEPT_CODE", "YYYYMM"],
            right_on=["DEPT_CODE", "YYYYMM"],
            how="inner",
        )

        y_true = merged[outcome].values.astype(float)
        y_pred = merged["pred_mean"].values.astype(float)
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, None)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        # Poisson NLL as a generic score (even if model is NB)
        nll = np.mean(y_pred - y_true * np.log(y_pred) + np.log(np.maximum(1, y_true)) * 0.0)

        rows.append(
            dict(
                outcome=outcome,
                fold=fold,
                cutoff_time=cutoff_time,
                horizon_months=horizon_months,
                rmse=rmse,
                mae=mae,
                poisson_nll=nll,
            )
        )

        print(
            f"Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, Poisson-NLL={nll:.3f}"
        )

    cv_results = pd.DataFrame(rows)
    return cv_results



def compute_aic_bic(
    idata,
    param_names=None,
    loglik_name: str = "y_like",
):
    """
    Compute approximate logL, AIC, BIC from an InferenceData object.

    Parameters
    ----------
    idata : arviz.InferenceData
    param_names : list[str] or None
        Names of parameters in idata.posterior to count for k.
        If None, all posterior data_vars are used.
    loglik_name : str
        Name of the observed variable in the log_likelihood group.

    Returns
    -------
    logL, AIC, BIC
    """

    # 1) log-likelihood: shape (chain, draw, obs)
    log_lik = idata.log_likelihood[loglik_name].values
    # sum over obs, then average over chain+draw
    logL_pointwise = log_lik.sum(axis=-1)        # (chain, draw)
    logL = logL_pointwise.mean().item()          # scalar

    # 2) number of observations
    n_obs = log_lik.shape[-1]

    # 3) number of parameters k
    posterior = idata.posterior
    all_params = list(posterior.data_vars.keys())

    if param_names is None:
        param_names = all_params
    else:
        # keep only those that actually exist
        param_names = [p for p in param_names if p in all_params]

    k = 0
    for name in param_names:
        arr = posterior[name].values
        # arr shape: (chain, draw, param_dims...)
        param_size = int(np.prod(arr.shape[2:]))  # ignore chain & draw
        k += param_size

    AIC = 2 * k - 2 * logL
    BIC = k * np.log(n_obs) - 2 * logL

    return float(logL), float(AIC), float(BIC)

def nb_car_gof(
    model,
    idata,
    y_obs: np.ndarray,
    var_name: str = "y_like",
    n_ppc_samples: int | None = 500,
):
    """
    Goodness-of-fit for NB-CAR using posterior predictive checks.

    Discrepancy statistics:
    1) Chi-square-like statistic:
           T_chi2(y) = sum_i (y_i - mu_i)^2 / Var_i
       where mu_i and Var_i are from posterior predictive.
    2) Number of zeros.

    For each statistic T, we compute:
        p_B = P( T(y_rep) >= T(y_obs) | data )
    via posterior predictive draws.

    Parameters
    ----------
    model : pm.Model
        Fitted NB-CAR PyMC model.
    idata : arviz.InferenceData
        Output of pm.sample(..., return_inferencedata=True).
    y_obs : np.ndarray
        Observed counts (1D, same order as model's "y").
    var_name : str
        Name of likelihood RV, e.g. "y_like".
    n_ppc_samples : int or None
        Number of posterior predictive draws to use (flattened over chains).
        If None, uses all available samples.

    Returns
    -------
    results : dict
        {
            "T_chi2_obs": float,
            "T_chi2_rep_mean": float,
            "p_chi2": float,
            "T_zero_obs": int,
            "T_zero_rep_mean": float,
            "p_zero": float
        }
    """

    y_obs = np.asarray(y_obs, dtype=float)

    # 1. Posterior predictive draws
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[var_name],
            return_inferencedata=True,
            progressbar=False,
        )

    y_ppc = ppc.posterior_predictive[var_name].values  # (chain, draw, obs)
    S_chain, S_draw, N_obs = y_ppc.shape
    y_ppc_flat = y_ppc.reshape(-1, N_obs)              # (S, obs)
    S_total = y_ppc_flat.shape[0]

    if (n_ppc_samples is not None) and (n_ppc_samples < S_total):
        idx = np.random.choice(S_total, size=n_ppc_samples, replace=False)
        y_ppc_flat = y_ppc_flat[idx]
        S_total = n_ppc_samples

    # 2. Predictive mean and variance per observation
    mu_hat = y_ppc_flat.mean(axis=0)
    var_hat = y_ppc_flat.var(axis=0, ddof=1) + 1e-8

    # Observed chi-square discrepancy
    T_chi2_obs = np.sum((y_obs - mu_hat) ** 2 / var_hat)

    # Replicated chi-square discrepancies
    T_chi2_rep = np.sum((y_ppc_flat - mu_hat) ** 2 / var_hat, axis=1)  # (S,)

    p_chi2 = float((T_chi2_rep >= T_chi2_obs).mean())

    # 3. Zero-count discrepancy
    T_zero_obs = int((y_obs == 0).sum())
    T_zero_rep = (y_ppc_flat == 0).sum(axis=1)  # (S,)
    T_zero_rep_mean = float(T_zero_rep.mean())
    p_zero = float((T_zero_rep >= T_zero_obs).mean())

    print("NB-CAR posterior predictive GOF:")
    print(f"  Chi-square discrepancy T_obs = {T_chi2_obs:.1f}")
    print(f"  mean T_rep = {T_chi2_rep.mean():.1f},  p_B (T_rep >= T_obs) = {p_chi2:.3f}")
    print("    -> p_B near 0.5 is ideal; very small or large suggests misfit.")
    print(f"  Zeros: T_obs = {T_zero_obs},  mean T_rep = {T_zero_rep_mean:.1f},  p_B = {p_zero:.3f}")
    print("    -> checks whether model matches sparsity / zero-inflation.")

    results = {
        "T_chi2_obs": float(T_chi2_obs),
        "T_chi2_rep_mean": float(T_chi2_rep.mean()),
        "p_chi2": p_chi2,
        "T_zero_obs": T_zero_obs,
        "T_zero_rep_mean": T_zero_rep_mean,
        "p_zero": p_zero,
    }
    return results

# -------------------------------------------------------------------
# 7. EXAMPLE USAGE (sketch)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Example: using your API to get panel data
    # from api import get_panel_data
    # df_polars = get_panel_data()

    # Here we assume df_polars already exists
    outcome = "NUMBER_OF_HOMICIDIO"
    df_pd, y, X_stdzd, meta = prepare_panel_for_st_car(
        df_polars=df,            # your Polars DataFrame
        outcome=outcome,
        x_cols=[f"S0{i}" for i in range(1, 9)],  # S01..S08
        k_neighbors=4,
    )

    model = build_st_car_poisson_model(df_pd, y, X_stdzd, meta)
    trace, idata = fit_st_car_model(model)

    # Diagnostics
    diagnostics_mcmc(idata)
    resid_pearson, mu_hat = diagnostics_ppc_residuals(model, trace, df_pd, y)
    diagnostics_spatial_temporal(resid_pearson, df_pd, meta)

    # Forecast 12 months ahead
    df_forecast = forecast_st_car(trace, df_pd, meta, horizon_months=12)
    print(df_forecast.head())



def compute_residuals_pearson(model, idata, y, var_name: str = "y_like"):
    """
    Compute posterior predictive Pearson residuals for a PyMC count model.

    Pearson residual for observation i:
        r_i = (y_i - mu_hat_i) / sqrt(Var_hat(y_i))

    where mu_hat_i and Var_hat(y_i) are estimated from posterior predictive draws.

    Parameters
    ----------
    model : pm.Model
        The fitted PyMC model (e.g., NB-CAR or ZINB-CAR).
    idata : arviz.InferenceData
        InferenceData returned by pm.sample(..., return_inferencedata=True).
    y : np.ndarray
        Observed counts, 1D array in the same order as the model's "y" data.
    var_name : str, default "y_like"
        Name of the observed likelihood RV in the model (usually "y_like").

    Returns
    -------
    resid_pearson : np.ndarray
        Pearson residuals, shape (N_obs,).
    mu_hat : np.ndarray
        Posterior predictive means, shape (N_obs,).
    var_hat : np.ndarray
        Posterior predictive variances, shape (N_obs,).
    """

    # 1. Posterior predictive sampling
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[var_name],
            return_inferencedata=True,
            progressbar=False,
        )

    # 2. Extract posterior predictive samples
    # ppc.posterior_predictive[var_name] has shape:
    #   (chain, draw, obs)   in most PyMC versions
    y_ppc = ppc.posterior_predictive[var_name].values

    # Flatten chain & draw dimensions → (samples, obs)
    if y_ppc.ndim == 3:
        # (chain, draw, obs)
        y_ppc_flat = y_ppc.reshape(-1, y_ppc.shape[-1])
    elif y_ppc.ndim == 2:
        # (draw, obs) – rare, but handle it
        y_ppc_flat = y_ppc
    else:
        raise ValueError(
            f"Unexpected PPC shape {y_ppc.shape}; expected 2 or 3 dimensions."
        )

    # 3. Posterior predictive mean & variance per observation
    mu_hat = y_ppc_flat.mean(axis=0)                  # (N_obs,)
    var_hat = y_ppc_flat.var(axis=0, ddof=1) + 1e-8   # (N_obs,) + jitter

    # 4. Pearson residuals
    y = np.asarray(y, dtype=float)
    resid_pearson = (y - mu_hat) / np.sqrt(var_hat)

    return resid_pearson, mu_hat, var_hat
