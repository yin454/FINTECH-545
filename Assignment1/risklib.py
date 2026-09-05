import numpy as np
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA


def aicc(log_likelihood, k, n):
    """
    Corrected Akaike Information Criterion.

    Parameters
    ----------
    log_likelihood : float
        Maximized log-likelihood of the fitted model.
    k : int
        Number of fitted parameters.
    n : int
        Number of observations.
    """
    aic = 2 * k - 2 * log_likelihood
    correction = (2 * k * (k + 1)) / (n - k - 1)
    return aic + correction


def sample_moments(x):
    """
    Return the first four sample moments used in Assignment 1.

    Conventions:
    - variance uses ddof=1
    - skewness uses the bias-corrected sample estimator
    - kurtosis is reported as excess kurtosis
    """
    x = np.asarray(x)

    return {
        "mean": np.mean(x),
        "variance": np.var(x, ddof=1),
        "skewness": stats.skew(x, bias=False),
        "excess_kurtosis": stats.kurtosis(x, fisher=True, bias=False),
    }


def fit_normal_regression(x, y):
    """
    Maximum-likelihood regression with Normal errors:

        y = alpha + beta*x + error
        error ~ Normal(0, sigma)

    Returns alpha, beta, sigma, log-likelihood, and AICc.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)

    # OLS values provide stable starting values.
    X = np.column_stack([np.ones(n), x])
    beta_start = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_start = y - X @ beta_start
    sigma_start = np.std(resid_start, ddof=0)

    def negative_log_likelihood(theta):
        alpha, beta, log_sigma = theta
        sigma = np.exp(log_sigma)

        residuals = y - (alpha + beta * x)

        return -np.sum(
            stats.norm.logpdf(
                residuals,
                loc=0.0,
                scale=sigma
            )
        )

    result = minimize(
        negative_log_likelihood,
        x0=[
            beta_start[0],
            beta_start[1],
            np.log(sigma_start),
        ],
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
    """
    Maximum-likelihood regression with Student-t errors:

        y = alpha + beta*x + error
        error ~ t(nu, location=0, scale=sigma)

    sigma is the t scale parameter, not the standard deviation.
    nu is the fitted degrees of freedom.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)

    # Use OLS values as starting values for alpha and beta.
    X = np.column_stack([np.ones(n), x])
    beta_start = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_start = y - X @ beta_start
    sigma_start = np.std(resid_start, ddof=0)

    def negative_log_likelihood(theta):
        alpha, beta, log_sigma, log_nu = theta

        sigma = np.exp(log_sigma)
        nu = np.exp(log_nu)

        residuals = y - (alpha + beta * x)

        return -np.sum(
            stats.t.logpdf(
                residuals,
                df=nu,
                loc=0.0,
                scale=sigma
            )
        )

    # Try several starting values for nu because the likelihood can be flat.
    nu_starts = [2, 3, 4, 5, 8, 15, 30]
    results = []

    for nu_start in nu_starts:
        result = minimize(
            negative_log_likelihood,
            x0=[
                beta_start[0],
                beta_start[1],
                np.log(sigma_start),
                np.log(nu_start),
            ],
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
    """
    Estimate the two-variable multivariate Normal conditional model
    for x2 given x1.

    Returns:
    - sample means
    - sample covariance matrix
    - regression/conditional mean coefficient
    - conditional variance
    - remaining variance factor
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    mu1 = np.mean(x1)
    mu2 = np.mean(x2)

    covariance = np.cov(x1, x2, ddof=1)

    sigma11 = covariance[0, 0]
    sigma12 = covariance[0, 1]
    sigma21 = covariance[1, 0]
    sigma22 = covariance[1, 1]

    beta = sigma21 / sigma11

    conditional_variance = (
        sigma22
        - sigma21 * (1.0 / sigma11) * sigma12
    )

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
    """
    Fit AR(1)-AR(3) and MA(1)-MA(3) with a constant.

    statsmodels reports AICc directly for ARIMA fits.
    """
    x = np.asarray(x)

    fits = {}

    for p in [1, 2, 3]:
        model = ARIMA(
            x,
            order=(p, 0, 0),
            trend="c"
        ).fit()

        fits[f"AR({p})"] = model

    for q in [1, 2, 3]:
        model = ARIMA(
            x,
            order=(0, 0, q),
            trend="c"
        ).fit()

        fits[f"MA({q})"] = model

    return fits
