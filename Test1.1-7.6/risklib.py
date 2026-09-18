
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# =========================================================
# 1. Missing-data covariance / correlation
# =========================================================

def covariance_matrix(data, skip_miss=True):
    """
    Covariance matrix with either listwise or pairwise deletion.

    skip_miss=True:
        Drop any row containing a missing value, then calculate covariance.
    skip_miss=False:
        Use pairwise available observations (pandas default behavior).
    """
    df = pd.DataFrame(data).copy()
    if skip_miss:
        df = df.dropna()
    return df.cov()


def correlation_matrix(data, skip_miss=True):
    """
    Correlation matrix with either listwise or pairwise deletion.
    """
    df = pd.DataFrame(data).copy()
    if skip_miss:
        df = df.dropna()
    return df.corr()


# =========================================================
# 2. Exponentially weighted covariance / correlation
# =========================================================

def exp_weights(n, lambd):
    """
    Normalized exponentially decaying weights for observations ordered
    from oldest to newest. The newest observation receives the largest weight.
    """
    powers = np.arange(n - 1, -1, -1)
    w = (1.0 - lambd) * (lambd ** powers)
    return w / w.sum()


def ew_covariance(data, lambd):
    """
    Exponentially weighted covariance matrix using normalized EW weights.
    """
    df = pd.DataFrame(data)
    x = df.to_numpy(dtype=float)
    w = exp_weights(len(df), lambd)

    mu = np.sum(x * w[:, None], axis=0)
    centered = x - mu
    cov = (centered * w[:, None]).T @ centered

    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def ew_correlation(data, lambd):
    """
    Exponentially weighted correlation matrix.
    """
    cov = ew_covariance(data, lambd)
    sd = np.sqrt(np.diag(cov))
    corr = cov.to_numpy() / np.outer(sd, sd)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def ew_covariance_mixed(data, variance_lambda=0.97, correlation_lambda=0.94):
    """
    Covariance matrix using EW variances from variance_lambda
    and EW correlations from correlation_lambda.
    """
    var_cov = ew_covariance(data, variance_lambda)
    sd = np.sqrt(np.diag(var_cov))
    corr = ew_correlation(data, correlation_lambda).to_numpy()
    cov = corr * np.outer(sd, sd)
    return pd.DataFrame(cov, index=var_cov.index, columns=var_cov.columns)


# =========================================================
# 3. PSD repair
# =========================================================

def _as_array_with_labels(data):
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float), list(data.columns)
    return np.asarray(data, dtype=float), None


def _restore_matrix(a, labels):
    if labels is None:
        return a
    return pd.DataFrame(a, index=labels, columns=labels)


def _cov_to_corr(cov):
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return corr, sd


def _corr_to_cov(corr, sd):
    return corr * np.outer(sd, sd)


def near_psd(data, epsilon=0.0):
    """
    Rebonato/Jaeckel-style near-PSD repair by clipping negative eigenvalues.
    Works for correlation or covariance matrices.
    """
    A, labels = _as_array_with_labels(data)
    is_corr = np.allclose(np.diag(A), 1.0)

    if is_corr:
        C = A.copy()
        sd = None
    else:
        C, sd = _cov_to_corr(A)

    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, epsilon)

    # Rescale so the repaired matrix keeps unit diagonal.
    scale = 1.0 / np.sqrt((vecs ** 2) @ vals)
    B = np.diag(scale) @ vecs @ np.diag(np.sqrt(vals))
    repaired_corr = B @ B.T

    result = repaired_corr if is_corr else _corr_to_cov(repaired_corr, sd)
    return _restore_matrix(result, labels)


def _project_psd(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)
    return vecs @ np.diag(vals) @ vecs.T


def higham_nearest_psd(data, tol=1e-9, max_iter=1000):
    """
    Higham nearest correlation-matrix algorithm using alternating projections
    with Dykstra's correction. Covariance inputs are converted to correlations,
    repaired, then rescaled back.
    """
    A, labels = _as_array_with_labels(data)
    is_corr = np.allclose(np.diag(A), 1.0)

    if is_corr:
        C = A.copy()
        sd = None
    else:
        C, sd = _cov_to_corr(A)

    Y = C.copy()
    delta_s = np.zeros_like(C)
    prev_norm = np.inf

    for _ in range(max_iter):
        R = Y - delta_s
        X = _project_psd(R)
        delta_s = X - R

        Y = X.copy()
        np.fill_diagonal(Y, 1.0)

        cur_norm = np.linalg.norm(Y - C, ord="fro")
        if abs(cur_norm - prev_norm) < tol:
            break
        prev_norm = cur_norm

    result = Y if is_corr else _corr_to_cov(Y, sd)
    return _restore_matrix(result, labels)


# =========================================================
# 4. Cholesky for PSD matrices
# =========================================================

def chol_psd(data, tol=1e-8):
    """
    Lower-triangular Cholesky-like factor for PSD matrices.

    Returns L such that L @ L.T ~= A.
    Small negative diagonal residuals caused by floating point error
    are treated as zero.
    """
    A, labels = _as_array_with_labels(data)
    n = A.shape[0]
    root = np.zeros_like(A)

    for j in range(n):
        s = np.dot(root[j, :j], root[j, :j])
        temp = A[j, j] - s

        if temp < 0 and abs(temp) <= tol:
            temp = 0.0
        if temp < 0:
            raise ValueError("Matrix is not positive semi-definite.")

        root[j, j] = np.sqrt(temp)

        if root[j, j] <= tol:
            continue

        for i in range(j + 1, n):
            s = np.dot(root[i, :j], root[j, :j])
            root[i, j] = (A[i, j] - s) / root[j, j]

    return _restore_matrix(root, labels)


# =========================================================
# 5. Simulation
# =========================================================

def simulate_normal(cov, n_sim=100000, mean=None, seed=None):
    """
    Multivariate normal simulation using the PSD-tolerant Cholesky factor.
    """
    A, labels = _as_array_with_labels(cov)
    n = A.shape[0]

    if mean is None:
        mean = np.zeros(n)
    mean = np.asarray(mean, dtype=float)

    L = chol_psd(A)
    if isinstance(L, pd.DataFrame):
        L = L.to_numpy()

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sim, n))
    sim = Z @ L.T + mean

    if labels is None:
        return sim
    return pd.DataFrame(sim, columns=labels)


def simulate_pca(cov, n_sim=100000, explained_variance=1.0, mean=None, seed=None):
    """
    PCA-based multivariate normal simulation.

    Keeps the minimum number of principal components needed to explain at least
    `explained_variance` of total positive variance.
    """
    A, labels = _as_array_with_labels(cov)
    n = A.shape[0]

    if mean is None:
        mean = np.zeros(n)
    mean = np.asarray(mean, dtype=float)

    vals, vecs = np.linalg.eigh(A)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    keep_positive = vals > 1e-12
    vals = vals[keep_positive]
    vecs = vecs[:, keep_positive]

    cumulative = np.cumsum(vals) / np.sum(vals)
    k = np.searchsorted(cumulative, explained_variance, side="left") + 1
    k = min(k, len(vals))

    vals = vals[:k]
    vecs = vecs[:, :k]
    B = vecs @ np.diag(np.sqrt(vals))

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sim, k))
    sim = Z @ B.T + mean

    if labels is None:
        return sim
    return pd.DataFrame(sim, columns=labels)


# =========================================================
# 6. Returns
# =========================================================

def calculate_returns(data, method="arithmetic", date_col="Date"):
    """
    Calculate arithmetic or log returns for every numeric price column.

    method:
        "arithmetic" -> P_t / P_{t-1} - 1
        "log"        -> ln(P_t / P_{t-1})
    """
    df = pd.DataFrame(data).copy()

    if date_col in df.columns:
        dates = df[date_col].copy()
        prices = df.drop(columns=[date_col])
    else:
        dates = None
        prices = df

    if method.lower() == "arithmetic":
        ret = prices.pct_change(fill_method=None)
    elif method.lower() == "log":
        ret = np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'arithmetic' or 'log'")

    ret = ret.iloc[1:].reset_index(drop=True)

    if dates is not None:
        ret.insert(0, date_col, dates.iloc[1:].reset_index(drop=True))

    return ret


# =========================================================
# 7. Distribution fitting / regression
# =========================================================

def fit_normal(data):
    """
    Fit Normal distribution using sample mean and sample standard deviation.
    Returns (mu, sigma).
    """
    x = np.asarray(data, dtype=float).reshape(-1)
    return np.mean(x), np.std(x, ddof=1)


def fit_t(data):
    """
    Fit Student-t distribution by maximum likelihood.
    Returns (mu, sigma, nu).
    """
    x = np.asarray(data, dtype=float).reshape(-1)
    nu, mu, sigma = stats.t.fit(x)
    return mu, sigma, nu


def fit_t_regression(X, y):
    """
    Student-t regression:
        y = alpha + X beta + epsilon
        epsilon ~ t(nu, loc=0, scale=sigma)

    Returns:
        mu, sigma, nu, alpha, beta
    where mu is fixed at 0 for the regression error distribution.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X[:, None]

    n, p = X.shape
    X_ols = np.column_stack([np.ones(n), X])
    coef0 = np.linalg.lstsq(X_ols, y, rcond=None)[0]
    alpha0 = coef0[0]
    beta0 = coef0[1:]
    resid0 = y - (alpha0 + X @ beta0)
    sigma0 = max(np.std(resid0, ddof=1), 1e-8)
    nu0 = 5.0

    theta0 = np.r_[alpha0, beta0, np.log(sigma0), np.log(nu0 - 2.0)]

    def nll(theta):
        alpha = theta[0]
        beta = theta[1:1 + p]
        sigma = np.exp(theta[1 + p])
        nu = 2.0 + np.exp(theta[2 + p])
        resid = y - alpha - X @ beta
        return -np.sum(stats.t.logpdf(resid, df=nu, loc=0.0, scale=sigma))

    result = minimize(
        nll,
        theta0,
        method="BFGS",
        options={"maxiter": 10000, "gtol": 1e-10},
    )

    theta = result.x
    alpha = theta[0]
    beta = theta[1:1 + p]
    sigma = np.exp(theta[1 + p])
    nu = 2.0 + np.exp(theta[2 + p])

    return 0.0, sigma, nu, alpha, beta


def calculate_aicc_t(data):
    """
    Fit a Student-t distribution and calculate AICC.
    k=3 parameters: mu, sigma, nu.
    """
    x = np.asarray(data, dtype=float).reshape(-1)
    mu, sigma, nu = fit_t(x)

    log_likelihood = np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=sigma))
    n = len(x)
    k = 3

    aic = 2 * k - 2 * log_likelihood
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def fit_nig_moments(data):
    """
    Fit a Normal Inverse Gaussian distribution by method of moments.

    Returns (mu, alpha, beta, delta).
    """
    x = np.asarray(data, dtype=float).reshape(-1)

    m = np.mean(x)
    var = np.var(x, ddof=1)
    skew = stats.skew(x, bias=True)
    ex_kurt = stats.kurtosis(x, fisher=True, bias=True)

    denom = 3.0 * ex_kurt - 4.0 * skew**2
    if denom <= 0:
        raise ValueError("Sample moments do not imply valid NIG parameters.")

    rho2 = skew**2 / denom
    rho = np.sign(skew) * np.sqrt(rho2)
    q = 9.0 / denom

    alpha = np.sqrt(q / (var * (1.0 - rho2) ** 2))
    beta = rho * alpha
    delta = q / (alpha * np.sqrt(1.0 - rho2))
    mu = m - delta * rho / np.sqrt(1.0 - rho2)

    return mu, alpha, beta, delta


def fit_nig_mle(data):
    """
    Fit a Normal Inverse Gaussian distribution by maximum likelihood.

    scipy.stats.norminvgauss uses:
        a = alpha * delta
        b = beta  * delta
        loc = mu
        scale = delta

    Returns (mu, alpha, beta, delta).
    """
    x = np.asarray(data, dtype=float).reshape(-1)
    a, b, mu, delta = stats.norminvgauss.fit(x)
    alpha = a / delta
    beta = b / delta
    return mu, alpha, beta, delta
