from models.transformations import coordinates
import numpy as np

# ---------------------------
# 2. Build C (relative influences)
# ---------------------------
def build_influence_matrix_C(
    latitudes:np.array,
    longitudes:np.array,
    populations:np.array=None,
    radius_km:float=250.0,
    d0:float=100.0,
    alpha:float=1.0,
    beta:float=1.0
) -> np.ndarray:
    """
    Build row-stochastic relative influence matrix C based on geography (and optionally population).

    Parameters
    ----------
    latitudes, longitudes : array-like of length N
        Coordinates of Colombian municipalities/cities.
    populations : array-like of length N or None
        Populations (optional but recommended).
    radius_km : float
        Only neighbours within this radius are considered.
    d0 : float
        Decay lengthscale used in exp(-d_ij / d0).
    alpha, beta : float
        Population exponents for gravity-type interaction.

    Returns
    -------
    C : (N, N) ndarray
        Relative influence matrix with zero diagonal and row sums 0 or 1.
    """

    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    N = len(longitudes)

    D = coordinates.distance_matrix_km(latitudes, longitudes)
    C = np.zeros((N, N), dtype=float)

    for i in range(N):
        # mask neighbours within radius (exclude self)
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        mask &= (D[i, :] <= radius_km)

        if not np.any(mask):
            # isolated node: no interpersonal influence
            continue

        dij = D[i, mask]
        if populations is not None:
            pj = populations[mask]
            # gravity-like weight
            # (pop_i**alpha can be factored out and hence omitted for row-normalization)
            raw = pj**beta * np.exp(-dij / d0)
        else:
            raw = np.exp(-dij / d0)

        # normalise
        raw_sum = raw.sum()
        if raw_sum == 0:
            continue

        C[i, mask] = raw / raw_sum

    # enforce zero diagonal explicitly
    np.fill_diagonal(C, 0.0)
    return C

# ---------------------------
# 3. Build W = AC + I - A
# ---------------------------
def build_W(C: np.ndarray, a):
    """
    Build Friedkin influence matrix W from relative influences C and susceptibilities a.

    Parameters
    ----------
    C : (N, N) ndarray
        Relative influence matrix (zero diagonal, row-normalized).
    a : (N,) array-like
        Susceptibility parameters 0 <= a_i <= 1.

    Returns
    -------
    W : (N, N) ndarray
        Influence matrix, row-stochastic.
    """
    C = np.asarray(C, dtype=float)
    a = np.asarray(a, dtype=float)
    N = C.shape[0]
    assert C.shape == (N, N)
    assert a.shape == (N,)

    A = np.diag(a)
    I = np.eye(N)
    W = A @ C + I - A  # W = AC + I - A

    # small numerical clean-up
    W[W < 0] = 0.0
    row_sums = W.sum(axis=1, keepdims=True)
    # avoid division by zero: if a row-sum is 0, leave it as-is (should not happen if a_i in [0,1])
    mask = row_sums.squeeze() > 0
    W[mask] /= row_sums[mask]
    return W







def build_W_scalar_a(C, a):
    """
    Version of build_W when a is a scalar: a_i = a for all i.
    """
    N = C.shape[0]
    A = np.diag(np.full(N, a))
    I = np.eye(N)
    W = A @ C + I - A
    # clean up small negatives and re-normalize rows
    W[W < 0] = 0.0
    row_sums = W.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    W[mask] /= row_sums[mask]
    return W

def build_baseline_from_early_years(Y_hist, K=5):
    """
    Time-invariant baseline: average conflict in first K years.
    Y_hist: shape (T, N)
    """
    return Y_hist[:K].mean(axis=0)

def simulate_one_step(Y_t, W, a, y_star_t):
    """
    One-step Friedkin-Johnsen update using scalar a and time-varying anchor y_star_t.
    """
    N = len(Y_t)
    A = np.diag(np.full(N, a))
    I = np.eye(N)
    return A @ W @ Y_t + (I - A) @ y_star_t

def fit_a_scalar(Y_hist, C, y_star, a_grid=None):
    """
    Fit scalar susceptibility a by minimizing squared error over time:
      Y_{t+1} ~ a W(a) Y_t + (1-a) y_star
    where W depends on a only via formula W = A C + I - A.
    In practice, W is nearly independent of a's scale here, but we recompute to be consistent.

    Parameters
    ----------
    Y_hist : (T, N) ndarray
        Historical conflict data.
    C : (N, N) ndarray
        Relative influence matrix.
    y_star : (N,) ndarray
        Time-invariant anchor.
    a_grid : array-like or None
        Candidate values for a. If None, use np.linspace(0, 0.99, 50).

    Returns
    -------
    best_a : float
        Value of a that minimizes squared error.
    """
    Y_hist = np.asarray(Y_hist)
    T, N = Y_hist.shape

    if a_grid is None:
        a_grid = np.linspace(0.0, 0.99, 50)

    best_a = None
    best_err = np.inf

    for a in a_grid:
        W = build_W_scalar_a(C, a)
        A = np.diag(np.full(N, a))
        I = np.eye(N)

        err = 0.0
        for t in range(T - 1):
            Y_t = Y_hist[t]
            Y_tp1_pred = A @ W @ Y_t + (I - A) @ y_star
            diff = Y_hist[t + 1] - Y_tp1_pred
            err += np.dot(diff, diff)  # squared L2 norm

        if err < best_err:
            best_err = err
            best_a = a

    return best_a, best_err

def simulate_with_history(Y_hist, C, y_star, a, steps_ahead=10):
    """
    Use last historical state as starting point and simulate forward.

    Parameters
    ----------
    Y_hist : (T, N)
        Historical conflict data, used only for starting state here.
    C : (N, N)
        Relative influence matrix.
    y_star : (N,)
        Time-invariant anchor.
    a : float
        Scalar susceptibility.
    steps_ahead : int
        Number of steps to simulate beyond last historical year.

    Returns
    -------
    Y_future : (steps_ahead, N)
        Simulated future trajectories.
    """
    Y_hist = np.asarray(Y_hist)
    T, N = Y_hist.shape
    Y_t = Y_hist[-1].copy()

    W = build_W_scalar_a(C, a)
    A = np.diag(np.full(N, a))
    I = np.eye(N)

    Y_future = np.zeros((steps_ahead, N))
    for h in range(steps_ahead):
        Y_t = A @ W @ Y_t + (I - A) @ y_star
        Y_future[h, :] = Y_t
    return Y_future