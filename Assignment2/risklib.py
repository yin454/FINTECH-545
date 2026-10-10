"""risklib.py

Combined FINTECH 545 risk library.

This file consolidates the functions used for Assignment 1, Tests 1.1-7.6,
and Assignment 2. Existing functions are retained where possible and new
Assignment 2 helpers are added at the end.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from statsmodels.tsa.arima.model import ARIMA


# =========================================================
# General information criteria / moments
# =========================================================

def aicc(log_likelihood, k, n):
    """Corrected Akaike Information Criterion."""
    aic = 2 * k - 2 * log_likelihood
    correction = (2 * k * (k + 1)) / (n - k - 1)
    return aic + correction


def bic(log_likelihood, k, n):
    """Bayesian Information Criterion."""
    return k * np.log(n) - 2 * log_likelihood


def sample_moments(x):
    """
    First four sample moments.

    Conventions:
    - variance uses ddof=1
    - skewness is bias-corrected
    - kurtosis is excess kurtosis and bias-corrected
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return {
        "mean": np.mean(x),
        "variance": np.var(x, ddof=1),
        "skewness": stats.skew(x, bias=False),
        "excess_kurtosis": stats.kurtosis(x, fisher=True, bias=False),
    }


# =========================================================
# Assignment 1 regression / conditional normal / AR-MA
# =========================================================

def fit_normal_regression(x, y):
    """MLE regression with Normal errors: y = alpha + beta*x + error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    X = np.column_stack([np.ones(n), x])
    beta_start = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_start = y - X @ beta_start
    sigma_start = max(np.std(resid_start, ddof=0), 1e-12)

    def negative_log_likelihood(theta):
        alpha, beta, log_sigma = theta
        sigma = np.exp(log_sigma)
        residuals = y - (alpha + beta * x)
        return -np.sum(stats.norm.logpdf(residuals, loc=0.0, scale=sigma))

    result = minimize(
        negative_log_likelihood,
        x0=[beta_start[0], beta_start[1], np.log(sigma_start)],
        method="BFGS",
    )

    alpha, beta, log_sigma = result.x
    sigma = np.exp(log_sigma)
    log_likelihood = -result.fun

    return {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "log_likelihood": log_likelihood,
        "aicc": aicc(log_likelihood, k=3, n=n),
    }


def fit_student_t_regression(x, y):
    """MLE regression with Student-t errors for one explanatory variable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    X = np.column_stack([np.ones(n), x])
    beta_start = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_start = y - X @ beta_start
    sigma_start = max(np.std(resid_start, ddof=0), 1e-12)

    def negative_log_likelihood(theta):
        alpha, beta, log_sigma, log_nu = theta
        sigma = np.exp(log_sigma)
        nu = np.exp(log_nu)
        residuals = y - (alpha + beta * x)
        return -np.sum(stats.t.logpdf(residuals, df=nu, loc=0.0, scale=sigma))

    results = []
    for nu_start in [2, 3, 4, 5, 8, 15, 30]:
        result = minimize(
            negative_log_likelihood,
            x0=[beta_start[0], beta_start[1], np.log(sigma_start), np.log(nu_start)],
            method="BFGS",
        )
        results.append(result)

    best = min(results, key=lambda r: r.fun)
    alpha, beta, log_sigma, log_nu = best.x
    sigma = np.exp(log_sigma)
    nu = np.exp(log_nu)
    log_likelihood = -best.fun

    return {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "nu": nu,
        "log_likelihood": log_likelihood,
        "aicc": aicc(log_likelihood, k=4, n=n),
    }


def conditional_normal_parameters(x1, x2):
    """Estimate the bivariate Normal conditional model for x2 given x1."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    mu1, mu2 = np.mean(x1), np.mean(x2)
    covariance = np.cov(x1, x2, ddof=1)
    sigma11, sigma12 = covariance[0, 0], covariance[0, 1]
    sigma21, sigma22 = covariance[1, 0], covariance[1, 1]
    beta = sigma21 / sigma11
    conditional_variance = sigma22 - sigma21 * sigma12 / sigma11
    variance_factor = conditional_variance / sigma22
    return {
        "mu1": mu1,
        "mu2": mu2,
        "covariance": covariance,
        "beta": beta,
        "conditional_variance": conditional_variance,
        "variance_factor": variance_factor,
    }


def fit_ar_ma_models(x):
    """Fit AR(1)-AR(3) and MA(1)-MA(3) with a constant."""
    x = np.asarray(x, dtype=float)
    fits = {}
    for p in [1, 2, 3]:
        fits[f"AR({p})"] = ARIMA(x, order=(p, 0, 0), trend="c").fit()
    for q in [1, 2, 3]:
        fits[f"MA({q})"] = ARIMA(x, order=(0, 0, q), trend="c").fit()
    return fits


# =========================================================
# Missing-data covariance / correlation
# =========================================================

def covariance_matrix(data, skip_miss=True):
    """Covariance matrix using listwise or pairwise deletion."""
    df = pd.DataFrame(data).copy()
    if skip_miss:
        df = df.dropna()
    return df.cov()


def correlation_matrix(data, skip_miss=True):
    """Correlation matrix using listwise or pairwise deletion."""
    df = pd.DataFrame(data).copy()
    if skip_miss:
        df = df.dropna()
    return df.corr()


def pairwise_observation_counts(data):
    """Count jointly observed rows for every pair of columns."""
    df = pd.DataFrame(data)
    cols = list(df.columns)
    out = pd.DataFrame(index=cols, columns=cols, dtype=int)
    for c1 in cols:
        for c2 in cols:
            out.loc[c1, c2] = df[[c1, c2]].dropna().shape[0]
    return out.astype(int)


# =========================================================
# Exponentially weighted covariance / correlation
# =========================================================

def exp_weights(n, lambd):
    """Normalized EW weights for observations ordered oldest to newest."""
    powers = np.arange(n - 1, -1, -1)
    w = (1.0 - lambd) * (lambd ** powers)
    return w / w.sum()


def ew_covariance(data, lambd):
    """Exponentially weighted covariance matrix."""
    df = pd.DataFrame(data)
    x = df.to_numpy(dtype=float)
    w = exp_weights(len(df), lambd)
    mu = np.sum(x * w[:, None], axis=0)
    centered = x - mu
    cov = (centered * w[:, None]).T @ centered
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def ew_correlation(data, lambd):
    """Exponentially weighted correlation matrix."""
    cov = ew_covariance(data, lambd)
    sd = np.sqrt(np.diag(cov))
    corr = cov.to_numpy() / np.outer(sd, sd)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def ew_covariance_mixed(data, variance_lambda=0.97, correlation_lambda=0.94):
    """EW covariance using one lambda for variances and another for correlations."""
    var_cov = ew_covariance(data, variance_lambda)
    sd = np.sqrt(np.diag(var_cov))
    corr = ew_correlation(data, correlation_lambda).to_numpy()
    cov = corr * np.outer(sd, sd)
    return pd.DataFrame(cov, index=var_cov.index, columns=var_cov.columns)


def ew_standard_deviation(x, lambd, demean=True):
    """Exponentially weighted standard deviation for a univariate series."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if demean:
        x = x - np.mean(x)
    w = exp_weights(len(x), lambd)
    return np.sqrt(np.sum(w * x**2))


def effective_sample_size(n, lambd):
    """Finite-history effective sample size: 1/sum(w_i^2)."""
    w = exp_weights(n, lambd)
    return 1.0 / np.sum(w**2)


def ew_half_life(lambd):
    """EWMA half-life in observations."""
    return np.log(0.5) / np.log(lambd)


# =========================================================
# PSD repair
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
    """Rebonato/Jaeckel-style near-PSD repair by eigenvalue clipping."""
    A, labels = _as_array_with_labels(data)
    is_corr = np.allclose(np.diag(A), 1.0)
    if is_corr:
        C, sd = A.copy(), None
    else:
        C, sd = _cov_to_corr(A)

    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, epsilon)
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
    """Higham nearest correlation matrix using alternating projections."""
    A, labels = _as_array_with_labels(data)
    is_corr = np.allclose(np.diag(A), 1.0)
    if is_corr:
        C, sd = A.copy(), None
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
# Cholesky / simulation
# =========================================================

def chol_psd(data, tol=1e-8):
    """Lower-triangular Cholesky-like factor for PSD matrices."""
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


def simulate_normal(cov, n_sim=100000, mean=None, seed=None):
    """Multivariate Normal simulation using PSD-tolerant Cholesky."""
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
    return sim if labels is None else pd.DataFrame(sim, columns=labels)


def simulate_pca(cov, n_sim=100000, explained_variance=1.0, mean=None, seed=None):
    """PCA-based multivariate Normal simulation."""
    A, labels = _as_array_with_labels(cov)
    n = A.shape[0]
    if mean is None:
        mean = np.zeros(n)
    mean = np.asarray(mean, dtype=float)
    vals, vecs = np.linalg.eigh(A)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    keep = vals > 1e-12
    vals, vecs = vals[keep], vecs[:, keep]
    cumulative = np.cumsum(vals) / np.sum(vals)
    k = np.searchsorted(cumulative, explained_variance, side="left") + 1
    k = min(k, len(vals))
    B = vecs[:, :k] @ np.diag(np.sqrt(vals[:k]))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sim, k))
    sim = Z @ B.T + mean
    return sim if labels is None else pd.DataFrame(sim, columns=labels)


# =========================================================
# Returns / distribution fitting
# =========================================================

def calculate_returns(data, method="arithmetic", date_col="Date"):
    """Calculate arithmetic or log returns for every price column."""
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


def fit_normal(data):
    """Fit Normal using sample mean and sample standard deviation (ddof=1)."""
    x = np.asarray(data, dtype=float).reshape(-1)
    return np.mean(x), np.std(x, ddof=1)


def fit_t(data):
    """Fit Student-t by maximum likelihood. Returns (mu, sigma, nu)."""
    x = np.asarray(data, dtype=float).reshape(-1)
    nu, mu, sigma = stats.t.fit(x)
    return mu, sigma, nu


def fit_t_regression(X, y):
    """Student-t regression for one or more explanatory variables."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    X_ols = np.column_stack([np.ones(n), X])
    coef0 = np.linalg.lstsq(X_ols, y, rcond=None)[0]
    alpha0, beta0 = coef0[0], coef0[1:]
    resid0 = y - (alpha0 + X @ beta0)
    sigma0 = max(np.std(resid0, ddof=1), 1e-8)
    theta0 = np.r_[alpha0, beta0, np.log(sigma0), np.log(3.0)]

    def nll(theta):
        alpha = theta[0]
        beta = theta[1:1 + p]
        sigma = np.exp(theta[1 + p])
        nu = 2.0 + np.exp(theta[2 + p])
        resid = y - alpha - X @ beta
        return -np.sum(stats.t.logpdf(resid, df=nu, loc=0.0, scale=sigma))

    result = minimize(nll, theta0, method="BFGS", options={"maxiter": 10000})
    theta = result.x
    alpha = theta[0]
    beta = theta[1:1 + p]
    sigma = np.exp(theta[1 + p])
    nu = 2.0 + np.exp(theta[2 + p])
    return 0.0, sigma, nu, alpha, beta


def calculate_aicc_t(data):
    """Fit Student-t and calculate AICc."""
    x = np.asarray(data, dtype=float).reshape(-1)
    mu, sigma, nu = fit_t(x)
    ll = np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=sigma))
    return aicc(ll, k=3, n=len(x))


def fit_nig_moments(data):
    """Fit a Normal Inverse Gaussian distribution by method of moments."""
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
    """Fit a Normal Inverse Gaussian distribution by maximum likelihood."""
    x = np.asarray(data, dtype=float).reshape(-1)
    a, b, mu, delta = stats.norminvgauss.fit(x)
    alpha = a / delta
    beta = b / delta
    return mu, alpha, beta, delta


# =========================================================
# Assignment 2: VaR / ES / marginal model selection
# =========================================================

def historical_var_es(pnl, alpha=0.05):
    """
    Historical VaR and ES from P&L.

    Sign convention: positive numbers denote losses. If the alpha-quantile
    P&L is positive, VaR can be negative, indicating a gain at that quantile.
    """
    pnl = np.asarray(pnl, dtype=float).reshape(-1)
    q = np.quantile(pnl, alpha)
    return -q, -np.mean(pnl[pnl <= q])


def normal_var(mean, std, alpha=0.05):
    """Normal VaR for a P&L distribution using the positive-loss convention."""
    return -(mean + stats.norm.ppf(alpha) * std)


def fit_normal_mle(data):
    """Normal MLE with log likelihood and AICc."""
    x = np.asarray(data, dtype=float).reshape(-1)
    mu = np.mean(x)
    sigma = np.std(x, ddof=0)
    ll = np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))
    return {"distribution": "normal", "mu": mu, "sigma": sigma,
            "log_likelihood": ll, "aicc": aicc(ll, 2, len(x))}


def fit_t_mle(data):
    """Student-t MLE with log likelihood and AICc."""
    x = np.asarray(data, dtype=float).reshape(-1)
    nu, mu, sigma = stats.t.fit(x)
    ll = np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=sigma))
    return {"distribution": "t", "mu": mu, "sigma": sigma, "nu": nu,
            "log_likelihood": ll, "aicc": aicc(ll, 3, len(x))}


def choose_normal_or_t(data):
    """Choose Normal or Student-t by AICc."""
    normal = fit_normal_mle(data)
    tfit = fit_t_mle(data)
    return (normal if normal["aicc"] <= tfit["aicc"] else tfit), normal, tfit


def fitted_cdf(x, fit):
    """Evaluate the CDF for a fit returned by fit_normal_mle / fit_t_mle."""
    x = np.asarray(x, dtype=float)
    if fit["distribution"] == "normal":
        return stats.norm.cdf(x, loc=fit["mu"], scale=fit["sigma"])
    return stats.t.cdf(x, df=fit["nu"], loc=fit["mu"], scale=fit["sigma"])


def fitted_ppf(u, fit):
    """Evaluate the inverse CDF for a fitted margin."""
    u = np.asarray(u, dtype=float)
    if fit["distribution"] == "normal":
        return stats.norm.ppf(u, loc=fit["mu"], scale=fit["sigma"])
    return stats.t.ppf(u, df=fit["nu"], loc=fit["mu"], scale=fit["sigma"])


# =========================================================
# Assignment 2: copulas
# =========================================================

def rank_uniforms(data):
    """Convert columns to scaled ranks r/(n+1), placing them strictly in (0,1)."""
    df = pd.DataFrame(data)
    return df.rank(method="average") / (len(df) + 1.0)


def kendall_tau_correlation(data):
    """Estimate a copula correlation matrix via rho = sin(pi*tau/2)."""
    df = pd.DataFrame(data)
    d = df.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau = stats.kendalltau(df.iloc[:, i], df.iloc[:, j]).statistic
            rho = np.sin(np.pi * tau / 2.0)
            R[i, j] = R[j, i] = rho
    return pd.DataFrame(R, index=df.columns, columns=df.columns)


def gaussian_copula_loglik(uniforms, corr, return_contributions=False):
    """Gaussian copula log likelihood."""
    U = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1 - 1e-12)
    R = np.asarray(corr, dtype=float)
    Z = stats.norm.ppf(U)
    ll_i = (stats.multivariate_normal.logpdf(Z, mean=np.zeros(Z.shape[1]), cov=R)
            - np.sum(stats.norm.logpdf(Z), axis=1))
    return ll_i if return_contributions else np.sum(ll_i)


def t_copula_loglik(uniforms, corr, nu, return_contributions=False):
    """Student-t copula log likelihood for fixed correlation and degrees of freedom."""
    U = np.clip(np.asarray(uniforms, dtype=float), 1e-12, 1 - 1e-12)
    R = np.asarray(corr, dtype=float)
    Q = stats.t.ppf(U, df=nu)
    ll_i = (stats.multivariate_t.logpdf(Q, loc=np.zeros(Q.shape[1]), shape=R, df=nu)
            - np.sum(stats.t.logpdf(Q, df=nu), axis=1))
    return ll_i if return_contributions else np.sum(ll_i)


def fit_t_copula(uniforms, corr, lower_nu=2.01, upper_nu=100.0):
    """Fit only nu for a t copula after the correlation matrix is fixed."""
    result = minimize_scalar(
        lambda nu: -t_copula_loglik(uniforms, corr, nu),
        bounds=(lower_nu, upper_nu),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return {"nu": result.x, "log_likelihood": -result.fun}


def simulate_gaussian_copula(corr, n_sim=100000, seed=None):
    """Simulate uniforms from a Gaussian copula."""
    R = np.asarray(corr, dtype=float)
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(np.zeros(R.shape[0]), R, size=n_sim)
    return stats.norm.cdf(Z)


def simulate_t_copula(corr, nu, n_sim=100000, seed=None):
    """Simulate uniforms from a Student-t copula."""
    R = np.asarray(corr, dtype=float)
    rng = np.random.default_rng(seed)
    Q = stats.multivariate_t.rvs(
        loc=np.zeros(R.shape[0]), shape=R, df=nu, size=n_sim, random_state=rng
    )
    return stats.t.cdf(Q, df=nu)


def t_copula_tail_dependence(rho, nu):
    """Symmetric upper/lower tail-dependence coefficient of a t copula."""
    x = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return 2.0 * stats.t.cdf(x, df=nu + 1.0)


# =========================================================
# Assignment 2: OLS factor model helper
# =========================================================

def ols_factor_regression(market_return, asset_return):
    """OLS of asset return on an intercept and one market return."""
    x = np.asarray(market_return, dtype=float).reshape(-1)
    y = np.asarray(asset_return, dtype=float).reshape(-1)
    X = np.column_stack([np.ones(len(y)), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coef
    return {
        "alpha": coef[0],
        "beta": coef[1],
        "residuals": residuals,
        "residual_std": np.std(residuals, ddof=1),
    }
