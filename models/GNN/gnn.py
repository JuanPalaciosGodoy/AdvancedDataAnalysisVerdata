# gnn_nb_spacetime.py

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

import numpy as np
import pandas as pd
import polars as pl

from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA

import torch



class SpatioTemporalGNN(nn.Module):
    """
    Time-aware Graph Neural Network with Negative Binomial likelihood.

    - Nodes: departments
    - Time: monthly steps
    - Graph: static spatial adjacency (edge_index)
    - Input X: (T, N, F) standardized features
    - Target Y: (T, N) counts (one outcome)

    Model:
        For each time t:
            h_t = GCN(X_t, edge_index)
        H = [h_1, ..., h_T]  # (T, N, gcn_hidden)
        H -> GRU over time -> Z_t (T, N, gru_hidden)
        log_mu_ti = linear(Z_ti)
        mu_ti = exp(log_mu_ti)  > 0

        Y_ti ~ NB(mu_ti, alpha)
        Var(Y_ti) = mu_ti + alpha * mu_ti^2

    alpha is a global, learned overdispersion parameter.

    Methods:
        - fit(X, Y, ...)
        - aic_bic(X, Y)
        - diagnostics(X, Y, time_index, node_index)
        - forecast(X, horizon_steps, use_last_covariates=True)
    """

    def __init__(
        self,
        n_nodes: int,
        in_channels: int,
        edge_index: torch.Tensor,
        gcn_hidden: int = 32,
        gru_hidden: int = 32,
        gru_layers: int = 1,
        device: str | torch.device = None,
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.in_channels = in_channels
        self.edge_index = edge_index
        self.gcn_hidden = gcn_hidden
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.node_embed = nn.Embedding(num_embeddings=n_nodes, embedding_dim=gcn_hidden)

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device

        # GCN layers
        self.gcn1 = GCNConv(in_channels, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)

        # GRU over time; batch dimension is nodes
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=False,  # (T, N, F)
        )

        # Output linear: from GRU hidden to log-mean
        self.out_linear = nn.Linear(gru_hidden, 1)

        # Global NB overdispersion parameter: alpha = exp(log_alpha) > 0
        self.log_alpha = nn.Parameter(torch.zeros(1))

        # Move to device
        self.to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        # Placeholder for training history / best state
        self._best_state = None
        self._fit_done = False

    # ------------------------------------------------------------------
    # Core forward pass: returns log_mu (T, N)
    # ------------------------------------------------------------------
    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        """
        X_seq: (T, N, F_in) on self.device
        Returns:
            log_mu: (T, N) tensor of NB mean log-parameters.
        """
        T, N, F_in = X_seq.shape
        assert N == self.n_nodes, "X_seq second dimension must be n_nodes"

        gcn_outputs = []
        for t in range(T):
            x_t = X_seq[t]  # (N, F_in)
            h = torch.relu(self.gcn1(x_t, self.edge_index))
            h = torch.relu(self.gcn2(h, self.edge_index))
            # add node embedding as a bias-like term
            h = h + self.node_embed(torch.arange(N, device=self.device))
            gcn_outputs.append(h)

        H = torch.stack(gcn_outputs, dim=0)  # (T, N, gcn_hidden)

        # GRU expects (T, batch=N, features=gcn_hidden)
        H_gru, _ = self.gru(H)  # (T, N, gru_hidden)

        # Output log_mu for each node-time
        log_mu = self.out_linear(H_gru).squeeze(-1)  # (T, N)

        return log_mu

    # ------------------------------------------------------------------
    # Negative Binomial log-likelihood
    # ------------------------------------------------------------------
    @staticmethod
    def _nb_loglik(y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Negative Binomial log-likelihood (sum over all y).

        Paramization:
            Var(Y) = mu + alpha * mu^2
            r = 1/alpha
            p = r / (r + mu)

        NB pmf: P(Y=y) = C(y+r-1,y) * p^r * (1-p)^y

        y, mu: (T, N)
        alpha: scalar > 0
        Returns:
            scalar log-likelihood (sum over all entries).
        """
        eps = 1e-8
        mu = torch.clamp(mu, min=eps)

        # alpha is scalar; broadcast
        alpha = torch.clamp(alpha, min=eps)
        r = 1.0 / alpha
        p = r / (r + mu)
        p = torch.clamp(p, min=eps, max=1 - eps)

        # log Gamma terms
        log_coeff = (
            torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
        )
        log_p = r * torch.log(p)
        log_1mp = y * torch.log(1 - p)

        log_prob = log_coeff + log_p + log_1mp
        return log_prob.sum()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.2,
        print_every: int = 20,
    ):
        """
        Fit the GNN using NB negative log-likelihood on a time split.

        Parameters
        ----------
        X : np.ndarray (T, N, F)
            Standardized features.
        Y : np.ndarray (T, N)
            Counts.
        n_epochs : int
        lr : float
        weight_decay : float
        val_fraction : float
            Fraction of last time steps used as validation.
        print_every : int
        """
        self.train()

        T, N, F = X.shape
        assert N == self.n_nodes, "Mismatch between n_nodes and X.shape[1]"

        # Train/val split by time
        val_T = int(T * val_fraction)
        train_T = T - val_T
        train_slice = slice(0, train_T)
        val_slice = slice(train_T, T)

        X_train = torch.tensor(X[train_slice], dtype=torch.float32, device=self.device)
        Y_train = torch.tensor(Y[train_slice], dtype=torch.float32, device=self.device)

        X_val = torch.tensor(X[val_slice], dtype=torch.float32, device=self.device)
        Y_val = torch.tensor(Y[val_slice], dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_nll = float("inf")
        best_state = None

        for epoch in range(1, n_epochs + 1):
            self.train()
            optimizer.zero_grad()

            log_mu_train = self.forward(X_train)  # (T_train, N)
            mu_train = torch.exp(log_mu_train)
            alpha = torch.exp(self.log_alpha)

            loglik_train = self._nb_loglik(Y_train, mu_train, alpha)
            loss_train = -loglik_train

            loss_train.backward()
            optimizer.step()

            # Validation
            self.eval()
            with torch.no_grad():
                log_mu_val = self.forward(X_val)
                mu_val = torch.exp(log_mu_val)
                loglik_val = self._nb_loglik(Y_val, mu_val, alpha)
                loss_val = -loglik_val

            if loss_val.item() < best_val_nll:
                best_val_nll = loss_val.item()
                best_state = {
                    "model": self.state_dict(),
                    "val_nll": best_val_nll,
                }

            if print_every and epoch % print_every == 0:
                print(
                    f"[Epoch {epoch}/{n_epochs}] "
                    f"Train NLL: {loss_train.item():.2f} | "
                    f"Val NLL: {loss_val.item():.2f} | "
                    f"alpha: {alpha.item():.4f}"
                )

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state["model"])
            self._best_state = best_state
        self._fit_done = True

    # ------------------------------------------------------------------
    # AIC / BIC
    # ------------------------------------------------------------------
    def aic_bic(self, X: np.ndarray, Y: np.ndarray):
        """
        Compute log-likelihood, AIC, BIC on a given dataset (e.g., training set).

        Parameters
        ----------
        X : np.ndarray (T, N, F)
        Y : np.ndarray (T, N)

        Returns
        -------
        logL, AIC, BIC
        """
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)

            log_mu = self.forward(X_t)
            mu = torch.exp(log_mu)
            alpha = torch.exp(self.log_alpha)

            logL = self._nb_loglik(Y_t, mu, alpha).item()

        n_obs = Y.size  # T * N
        k_params = sum(p.numel() for p in self.parameters())

        AIC = 2 * k_params - 2 * logL
        BIC = k_params * np.log(n_obs) - 2 * logL

        return logL, AIC, BIC

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        time_index: np.ndarray | None = None,
        node_index: np.ndarray | None = None,
        max_lag: int = 24,
    ):


        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
            log_mu = self.forward(X_t)
            mu = torch.exp(log_mu)

        mu_np = mu.cpu().numpy()
        y_np = Y_t.cpu().numpy()

        mu_flat = mu_np.ravel()
        y_flat = y_np.ravel()

        # ✅ use NB variance
        alpha = np.exp(self.log_alpha.detach().cpu().numpy())[0]
        var_nb = mu_flat + alpha * (mu_flat ** 2)
        var_nb = np.clip(var_nb, 1e-8, None)

        resid_pearson = (y_flat - mu_flat) / np.sqrt(var_nb)

        # 1) Residuals vs fitted
        plt.figure(figsize=(7, 5))
        plt.scatter(mu_flat, resid_pearson, alpha=0.3)
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Fitted mean (mu_hat)")
        plt.ylabel("Pearson residual (NB)")
        plt.title("SpatioTemporalGNN NB: residuals vs fitted")
        plt.show()

        # 2) QQ plot
        sm.qqplot(resid_pearson, line="45")
        plt.title("SpatioTemporalGNN NB: QQ plot of Pearson residuals")
        plt.show()

        # 3) Approx 95% NB predictive coverage (same as before)
        std_nb = np.sqrt(var_nb)
        lower = mu_flat - 1.96 * std_nb
        upper = mu_flat + 1.96 * std_nb
        lower = np.clip(lower, 0, None)
        covered = (y_flat >= lower) & (y_flat <= upper)
        coverage_rate = covered.mean()
        print(f"Approx. 95% NB predictive coverage: {coverage_rate:.3f}")

        # 4) Temporal ACF of mean residuals
        T, N = mu_np.shape
        resid_reshaped = resid_pearson.reshape(T, N)
        resid_time_mean = resid_reshaped.mean(axis=1)
        acf_vals = acf(resid_time_mean, fft=True, nlags=max_lag)

        plt.figure(figsize=(7, 4))
        plt.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        plt.xlabel("Lag (months)")
        plt.ylabel("ACF (mean residual)")
        plt.title("SpatioTemporalGNN NB: temporal ACF of mean residuals")
        plt.show()

        return {
            "resid_pearson": resid_pearson,
            "mu_flat": mu_flat,
            "coverage_rate": coverage_rate,
            "acf_time": acf_vals,
        }


    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------
    def forecast(
        self,
        X: np.ndarray,
        horizon_steps: int,
        use_last_covariates: bool = True,
    ):
        """
        Forecast future NB means (and optional samples) for each node.

        Simple approach:
        - Uses the last time step's covariates for all future steps
          if use_last_covariates=True (parallel to CAR pipeline).
        - Autoregressive in time through the GRU; we roll forward in one shot
          by building an extended X_seq with repeated last covariates.

        Parameters
        ----------
        X : np.ndarray (T, N, F)
            Historical standardized features.
        horizon_steps : int
            Number of future time steps to forecast.
        use_last_covariates : bool
            If True, reuse last observed covariates for all future steps.

        Returns
        -------
        mu_future : np.ndarray (H, N)
            Expected future means E[Y] for each future time and node.
        """
        self.eval()
        T, N, F = X.shape
        assert N == self.n_nodes, "Mismatch between n_nodes and X.shape[1]"

        if use_last_covariates:
            last_X = X[-1]  # (N, F)
            future_X = np.repeat(last_X[None, :, :], horizon_steps, axis=0)
        else:
            raise NotImplementedError(
                "For custom future covariates, build X_future externally and call forward() manually."
            )

        X_all = np.concatenate([X, future_X], axis=0)  # (T + H, N, F)

        with torch.no_grad():
            X_torch = torch.tensor(X_all, dtype=torch.float32, device=self.device)
            log_mu_all = self.forward(X_torch)
            mu_all = torch.exp(log_mu_all).cpu().numpy()  # (T+H, N)

        mu_future = mu_all[-horizon_steps:]  # (H, N)
        return mu_future



# -------------------------------------------------------------------
# 1. PREPARE PANEL FOR GNN
# -------------------------------------------------------------------

def prepare_panel_for_gnn(
    df_polars: pl.DataFrame,
    outcome: str,
    sector_cols=None,
    use_pca: bool = True,
    pca_var: float = 0.95,
    max_pcs: int | None = None,
    k_neighbors: int = 4,
    add_lags: bool = True,
    lag1: int = 1,
    lagS: int = 12,
    sqrt_offset: float = 0.5,
):
    """
    Prepare panel data for the NB SpatioTemporalGNN:
      - PCA on sector S0X columns
      - optional lagged target features (sqrt-transformed)
      - standardize features
      - build spatial graph (edge_index)
      - return X(T,N,F) and Y(T,N)
    """

    if sector_cols is None:
        sector_cols = [f"S0{i}" for i in range(1, 9)]

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

    dept_codes = np.sort(df_pd["DEPT_CODE"].unique())
    time_vals = np.sort(df_pd["YYYYMM"].unique())
    n_dept = len(dept_codes)
    n_time = len(time_vals)

    dept_to_idx = {d: i for i, d in enumerate(dept_codes)}
    time_to_idx = {t: i for i, t in enumerate(time_vals)}

    df_pd["dept_idx"] = df_pd["DEPT_CODE"].map(dept_to_idx)
    df_pd["time_idx"] = df_pd["YYYYMM"].map(time_to_idx)
    df_pd["month_idx"] = df_pd["YYYYMM"].dt.month - 1

    # --- outcome ---
    y_vec = df_pd[outcome].astype(float).values

    # ---------- PCA on sector columns ----------
    X_sector = df_pd[sector_cols].to_numpy().astype(float)

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

        components = pca.components_[:k]
        X_pcs = X_sector_std @ components.T

        X_design = X_pcs
        x_cols_effective = [f"PC{i+1}" for i in range(k)]

        pca_info = {
            "enabled": True,
            "sector_cols": sector_cols,
            "sector_mean": sector_mean,
            "sector_std": sector_std,
            "components": components,
            "n_components": k,
            "explained_variance_ratio": pca.explained_variance_ratio_[:k],
        }
    else:
        X_design = X_sector
        x_cols_effective = sector_cols
        pca_info = {"enabled": False}

    # ---------- add month-of-year feature ----------
    month_feat = df_pd["YYYYMM"].dt.month.values[:, None].astype(float)
    X_full = np.hstack([X_design, month_feat])
    x_cols_effective = x_cols_effective + ["month"]

    # ---------- add lagged target features ----------
    if add_lags:
        df_pd["_y_sqrt"] = np.sqrt(df_pd[outcome].astype(float).values + sqrt_offset)

        # sort properly before lags
        df_pd = df_pd.sort_values(["DEPT_CODE", "YYYYMM"])
        lag_features = []

        if lag1 is not None and lag1 > 0:
            df_pd[f"y_lag{lag1}"] = (
                df_pd.groupby("DEPT_CODE")["_y_sqrt"].shift(lag1)
            )
            lag_features.append(f"y_lag{lag1}")

        if lagS is not None and lagS > 0:
            df_pd[f"y_lag{lagS}"] = (
                df_pd.groupby("DEPT_CODE")["_y_sqrt"].shift(lagS)
            )
            lag_features.append(f"y_lag{lagS}")

        # re-align with X_full order
        df_pd = df_pd.sort_values(["DEPT_CODE", "YYYYMM"]).reset_index(drop=True)

        lag_mat = df_pd[lag_features].to_numpy().astype(float)
        # replace initial NaNs (where lags undefined) with 0
        lag_mat = np.nan_to_num(lag_mat, nan=0.0)

        X_full = np.hstack([X_full, lag_mat])
        x_cols_effective += lag_features

    # ---------- standardize final feature matrix ----------
    X_mean = X_full.mean(axis=0, keepdims=True)
    X_std = X_full.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_stdzd_vec = (X_full - X_mean) / X_std

    N_total, F = X_stdzd_vec.shape

    # ---------- reshape into (T, N, F) and (T, N) ----------
    X_stdzd = np.zeros((n_time, n_dept, F), dtype=float)
    Y = np.zeros((n_time, n_dept), dtype=float)

    for idx, row in df_pd.iterrows():
        t = row["time_idx"]
        d = row["dept_idx"]
        X_stdzd[t, d, :] = X_stdzd_vec[idx]
        Y[t, d] = y_vec[idx]

    # ---------- spatial graph ----------
    coords = (
        df_pd
        .drop_duplicates(subset=["DEPT_CODE"])
        .set_index("DEPT_CODE")[["DEPT_LAT", "DEPT_LON"]]
        .loc[dept_codes]
        .to_numpy()
    )

    W_sparse = kneighbors_graph(coords, n_neighbors=k_neighbors,
                                mode="connectivity", include_self=False)
    W = W_sparse.toarray().astype(float)
    W = np.maximum(W, W.T)

    edge_index = torch.tensor(
        np.vstack(np.where(W > 0)),
        dtype=torch.long,
    )

    meta = {
        "dept_codes": dept_codes,
        "time_vals": time_vals,
        "dept_to_idx": dept_to_idx,
        "time_to_idx": time_to_idx,
        "coords": coords,
        "X_cols": x_cols_effective,
        "X_mean": X_mean,
        "X_std": X_std,
        "pca": pca_info,
        "n_dept": n_dept,
        "n_time": n_time,
        "F": F,
        "outcome": outcome,
        "add_lags": add_lags,
        "lag1": lag1,
        "lagS": lagS,
        "sqrt_offset": sqrt_offset,
    }

    return X_stdzd, Y, edge_index, df_pd, meta



# -------------------------------------------------------------------
# 2. FIT NB-GNN PANEL
# -------------------------------------------------------------------

def fit_gnn_panel(
    X_stdzd: np.ndarray,
    Y: np.ndarray,
    edge_index: torch.Tensor,
    gcn_hidden: int = 32,
    gru_hidden: int = 32,
    gru_layers: int = 1,
    n_epochs: int = 200,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    print_every: int = 50,
):
    """
    Fit the NB SpatioTemporalGNN to the full panel and compute:
      - in-sample residuals
      - AIC / BIC

    Parameters
    ----------
    X_stdzd : np.ndarray (T,N,F)
    Y : np.ndarray (T,N)
    edge_index : torch.LongTensor (2,E)

    Returns
    -------
    model : SpatioTemporalGNN
    metrics_in : dict
        {"rmse", "mae", "poisson_nll", "logL", "AIC", "BIC"}
    resid_pearson : np.ndarray (T*N,)
    mu_flat : np.ndarray (T*N,)
    """

    T, N, F = X_stdzd.shape

    model = SpatioTemporalGNN(
        n_nodes=N,
        in_channels=F,
        edge_index=edge_index,
        gcn_hidden=gcn_hidden,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
    )

    model.fit(
        X_stdzd,
        Y,
        n_epochs=n_epochs,
        lr=lr,
        val_fraction=val_fraction,
        print_every=print_every,
    )

    # AIC/BIC on full data
    logL, AIC, BIC = model.aic_bic(X_stdzd, Y)

    # Fitted means
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_stdzd, dtype=torch.float32, device=model.device)
        log_mu = model(X_t)
        mu = torch.exp(log_mu).cpu().numpy()  # (T,N)

    mu_flat = mu.ravel()
    y_flat = Y.ravel().astype(float)

    # NB variance: Var(Y) = mu + alpha * mu^2
    alpha = float(torch.exp(model.log_alpha).detach().cpu().numpy())
    var_flat = mu_flat + alpha * (mu_flat**2) + 1e-8

    resid_pearson = (y_flat - mu_flat) / np.sqrt(var_flat)

    # In-sample metrics
    eps = 1e-8
    mu_clipped = np.clip(mu_flat, eps, None)

    rmse = np.sqrt(np.mean((y_flat - mu_clipped) ** 2))
    mae = np.mean(np.abs(y_flat - mu_clipped))
    # Poisson NLL as generic loss (for comparability)
    poisson_nll = np.mean(mu_clipped - y_flat * np.log(mu_clipped))

    metrics_in = {
        "rmse": rmse,
        "mae": mae,
        "poisson_nll": poisson_nll,
        "logL": logL,
        "AIC": AIC,
        "BIC": BIC,
        "alpha_nb": alpha,
    }

    return model, metrics_in, resid_pearson, mu_flat


# -------------------------------------------------------------------
# 3. ROLLING ORIGIN CV FOR NB-GNN
# -------------------------------------------------------------------

def rolling_cv_gnn(
    X_stdzd: np.ndarray,
    Y: np.ndarray,
    edge_index: torch.Tensor,
    time_vals: np.ndarray,
    n_folds: int = 3,
    horizon_months: int = 6,
    gcn_hidden: int = 32,
    gru_hidden: int = 32,
    gru_layers: int = 1,
    n_epochs: int = 150,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
):
    """
    Rolling-origin time-series cross-validation for NB-GNN.

    Parameters
    ----------
    X_stdzd : np.ndarray (T,N,F)
    Y : np.ndarray (T,N)
    edge_index : torch.LongTensor (2,E)
    time_vals : np.ndarray (T,)
        Datetime-like array for printing / reference.
    n_folds : int
    horizon_months : int
    ...

    Returns
    -------
    cv_results : pandas.DataFrame
        One row per fold with RMSE/MAE/NLL.
    """

    T, N, F = X_stdzd.shape
    rows = []

    # choose cut points in time (like CAR CV: in later part of series)
    fold_cut_indices = np.linspace(
        int(T * 0.4),
        T - horizon_months - 1,
        n_folds,
        dtype=int,
    )

    for fold, cut_idx in enumerate(fold_cut_indices, start=1):
        cutoff_time = time_vals[cut_idx]
        print(f"\n=== GNN CV fold {fold}/{n_folds}: train <= {cutoff_time} ===")

        T_train = cut_idx + 1
        T_test_end = min(T, T_train + horizon_months)

        X_train = X_stdzd[:T_train]
        Y_train = Y[:T_train]

        X_test = X_stdzd[T_train:T_test_end]
        Y_test = Y[T_train:T_test_end]

        # Fit new GNN on training slice
        model = SpatioTemporalGNN(
            n_nodes=N,
            in_channels=F,
            edge_index=edge_index,
            gcn_hidden=gcn_hidden,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
        )

        model.fit(
            X_train,
            Y_train,
            n_epochs=n_epochs,
            lr=lr,
            val_fraction=val_fraction,
            print_every=50,
        )

        # Predictions on test slice
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32, device=model.device)
            log_mu_test = model(X_t)
            mu_test = torch.exp(log_mu_test).cpu().numpy()  # (T_test_fold, N)

        y_true = Y_test.ravel().astype(float)
        mu_flat = mu_test.ravel()
        eps = 1e-8
        mu_clipped = np.clip(mu_flat, eps, None)

        # NB variance (for diagnostics if needed)
        alpha = float(torch.exp(model.log_alpha).detach().cpu().numpy())
        var_flat = mu_flat + alpha * (mu_flat**2) + 1e-8
        resid_pearson = (y_true - mu_flat) / np.sqrt(var_flat)

        rmse = np.sqrt(np.mean((y_true - mu_clipped) ** 2))
        mae = np.mean(np.abs(y_true - mu_clipped))
        poisson_nll = np.mean(mu_clipped - y_true * np.log(mu_clipped))

        rows.append(
            dict(
                fold=fold,
                cutoff_time=cutoff_time,
                horizon_months=horizon_months,
                rmse=rmse,
                mae=mae,
                poisson_nll=poisson_nll,
            )
        )

        print(
            f"GNN Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, Poisson-NLL={poisson_nll:.3f}"
        )

    cv_results = pd.DataFrame(rows)
    return cv_results


# -------------------------------------------------------------------
# 4. DIAGNOSTICS WRAPPER
# -------------------------------------------------------------------

def gnn_diagnostics_full(
    model: SpatioTemporalGNN,
    X_stdzd: np.ndarray,
    Y: np.ndarray,
    time_vals: np.ndarray | None = None,
    node_index: np.ndarray | None = None,
    max_lag: int = 24,
):
    """
    Run the GNN's internal diagnostics and return residuals + fitted means.
    """

    out = model.diagnostics(X_stdzd, Y, time_index=time_vals, node_index=node_index, max_lag=max_lag)
    resid_pearson = out["resid_pearson"]
    mu_flat = out["mu_flat"]
    return resid_pearson, mu_flat, out


