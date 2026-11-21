"""FINTECH 545 - Risk functions for Tests 8.1 through 13.9."""

import numpy as np
import pandas as pd
from scipy import stats


# =========================================================
# 8. Distribution fitting and VaR / ES
# =========================================================

def fit_normal(data):
    """Return (mean, sample standard deviation) for a normal distribution."""
    x = np.asarray(data, dtype=float).reshape(-1)
    return np.mean(x), np.std(x, ddof=1)


def fit_t(data):
    """Fit Student-t by maximum likelihood; return (mu, scale, nu)."""
    x = np.asarray(data, dtype=float).reshape(-1)
    nu, mu, sigma = stats.t.fit(x)
    return mu, sigma, nu


def normal_var(data, alpha=0.05):
    """Return (absolute VaR, VaR relative to the mean)."""
    mu, sigma = fit_normal(data)
    quantile = stats.norm.ppf(alpha, loc=mu, scale=sigma)
    return -quantile, mu - quantile


def t_var(data, alpha=0.05):
    """Return (absolute VaR, VaR relative to the mean) for a fitted t."""
    mu, sigma, nu = fit_t(data)
    quantile = stats.t.ppf(alpha, df=nu, loc=mu, scale=sigma)
    return -quantile, mu - quantile


def normal_es(data, alpha=0.05):
    """Return (absolute ES, ES relative to the mean) for a normal."""
    mu, sigma = fit_normal(data)
    z = stats.norm.ppf(alpha)
    tail_mean = mu - sigma * stats.norm.pdf(z) / alpha
    return -tail_mean, mu - tail_mean


def t_es(data, alpha=0.05):
    """Return (absolute ES, ES relative to the mean) for a fitted t."""
    mu, sigma, nu = fit_t(data)
    z = stats.t.ppf(alpha, df=nu)
    tail_mean = mu - sigma * (nu + z**2) / (nu - 1) * stats.t.pdf(z, df=nu) / alpha
    return -tail_mean, mu - tail_mean


def simulate_t(data, n_sim=100000, seed=1234):
    """Simulate from the fitted Student-t distribution with a reproducible seed."""
    mu, sigma, nu = fit_t(data)
    rng = np.random.default_rng(seed)
    return stats.t.rvs(df=nu, loc=mu, scale=sigma, size=n_sim, random_state=rng)


def simulated_var(simulations, alpha=0.05):
    """Calculate empirical VaR of simulated returns."""
    x = np.asarray(simulations, dtype=float).reshape(-1)
    quantile = np.quantile(x, alpha)
    return -quantile, np.mean(x) - quantile


def simulated_es(simulations, alpha=0.05):
    """Calculate empirical ES as the mean of the worst alpha fraction."""
    x = np.asarray(simulations, dtype=float).reshape(-1)
    quantile = np.quantile(x, alpha)
    tail_mean = np.mean(x[x <= quantile])
    return -tail_mean, np.mean(x) - tail_mean


# =========================================================
# 9. Copula portfolio VaR / ES
# =========================================================

def gaussian_copula_portfolio_risk(returns, portfolio, n_sim=300000, seed=1234, alpha=0.05):
    """Simulate Normal/t stock returns using a Gaussian copula.

    Dependence is estimated from Kendall's tau. Each stock's distribution
    is specified in the portfolio file. VaR/ES are positive dollar losses.
    """
    import pandas as pd

    returns = pd.DataFrame(returns)
    portfolio = pd.DataFrame(portfolio)
    stocks = portfolio["Stock"].tolist()

    # Kendall's tau -> copula correlation, for every stock pair.
    n_stocks = len(stocks)
    correlation = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            tau = stats.kendalltau(returns[stocks[i]], returns[stocks[j]]).statistic
            correlation[i, j] = np.sin(np.pi * tau / 2)
            correlation[j, i] = correlation[i, j]

    # Generate correlated uniforms from the Gaussian copula.
    rng = np.random.default_rng(seed)
    normal_draws = rng.multivariate_normal(np.zeros(n_stocks), correlation, size=n_sim)
    uniforms = stats.norm.cdf(normal_draws)

    results = []
    total_pnl = np.zeros(n_sim)
    total_value = 0.0

    for i, position in portfolio.reset_index(drop=True).iterrows():
        stock = position["Stock"]
        investment = float(position["Holding"] * position["Starting Price"])
        history = returns[stock].dropna().to_numpy()

        # Fit the required distribution, then invert its CDF.
        distribution = str(position["Distribution"]).strip().lower()
        if distribution == "normal":
            mu, sigma = fit_normal(history)
            simulated_returns = stats.norm.ppf(uniforms[:, i], loc=mu, scale=sigma)
        elif distribution in ("t", "student-t", "student's t"):
            mu, sigma, nu = fit_t(history)
            simulated_returns = stats.t.ppf(uniforms[:, i], df=nu, loc=mu, scale=sigma)
        else:
            raise ValueError(f"Unknown distribution for {stock}: {distribution}")

        pnl = investment * simulated_returns
        total_pnl += pnl
        total_value += investment

        cutoff = np.quantile(pnl, alpha)
        var = -cutoff
        es = -pnl[pnl <= cutoff].mean()
        results.append([stock, var, es, var / investment, es / investment])

    cutoff = np.quantile(total_pnl, alpha)
    total_var = -cutoff
    total_es = -total_pnl[total_pnl <= cutoff].mean()
    results.append(["Total", total_var, total_es, total_var / total_value, total_es / total_value])

    return pd.DataFrame(results, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])


# =========================================================
# 10. Risk parity and maximum Sharpe ratio
# =========================================================

def portfolio_risk_contributions(weights, covariance):
    """Return each asset's contribution to the portfolio standard deviation."""
    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    portfolio_sd = np.sqrt(weights @ covariance @ weights)
    return weights * (covariance @ weights) / portfolio_sd


def risk_parity_weights(covariance, risk_budgets=None):
    """Find nonnegative portfolio weights matching the target risk shares."""
    from scipy.optimize import minimize

    covariance = np.asarray(covariance, dtype=float)
    n = covariance.shape[0]

    if risk_budgets is None:
        risk_budgets = np.ones(n)
    risk_budgets = np.asarray(risk_budgets, dtype=float)
    risk_budgets = risk_budgets / risk_budgets.sum()

    def objective(weights):
        contributions = portfolio_risk_contributions(weights, covariance)
        risk_shares = contributions / contributions.sum()
        return np.sum((risk_shares - risk_budgets) ** 2)

    result = minimize(
        objective,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(1e-10, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-14, "maxiter": 5000},
    )
    if not result.success or objective(result.x) > 1e-10:
        raise RuntimeError("Risk parity optimization did not converge")
    return result.x


def max_sharpe_weights(means, covariance, rf=0.04, lower=0.0, upper=1.0):
    """Maximize (portfolio expected return - rf) / portfolio volatility."""
    from scipy.optimize import minimize

    means = np.asarray(means, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = len(means)

    def negative_sharpe(weights):
        portfolio_return = weights @ means
        portfolio_sd = np.sqrt(weights @ covariance @ weights)
        return -(portfolio_return - rf) / portfolio_sd

    result = minimize(
        negative_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(lower, upper)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-14, "maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError("Maximum Sharpe optimization did not converge")
    return result.x


# =========================================================
# 11. Ex-post return and volatility attribution
# =========================================================

def ex_post_attribution(returns, weights):
    """Attribute compounded return and daily volatility with drifting weights."""
    returns = pd.DataFrame(returns).astype(float)
    weights = np.asarray(weights, dtype=float).copy()
    daily_contributions = []
    portfolio_returns = []

    for _, daily_returns in returns.iterrows():
        r = daily_returns.to_numpy()
        contributions = weights * r
        portfolio_return = contributions.sum()
        daily_contributions.append(contributions)
        portfolio_returns.append(portfolio_return)

        # Buy-and-hold weights change as assets earn different returns.
        weights = weights * (1 + r) / (1 + portfolio_return)

    daily_contributions = np.array(daily_contributions)
    portfolio_returns = np.array(portfolio_returns)
    total_return = np.prod(1 + portfolio_returns) - 1
    linking = np.ones(len(portfolio_returns))
    valid = np.abs(portfolio_returns) > 1e-12
    linking[valid] = np.log1p(portfolio_returns[valid]) / portfolio_returns[valid]
    linking = linking / (np.log1p(total_return) / total_return)

    return_contributions = (daily_contributions * linking[:, None]).sum(axis=0)
    portfolio_sd = np.std(portfolio_returns, ddof=1)
    volatility_contributions = (
        np.cov(daily_contributions.T, portfolio_returns, ddof=1)[:-1, -1]
        / portfolio_sd
    )
    asset_total_returns = np.prod(1 + returns.to_numpy(), axis=0) - 1

    result = pd.DataFrame(
        [asset_total_returns, return_contributions, volatility_contributions],
        index=["TotalReturn", "Return Attribution", "Vol Attribution"],
        columns=returns.columns,
    )
    result.insert(0, "Value", result.index)
    result["Portfolio"] = [total_return, return_contributions.sum(), portfolio_sd]
    return result.reset_index(drop=True)


def ex_post_factor_attribution(stock_returns, factor_returns, beta, weights):
    """Attribute portfolio return and volatility to factors and unexplained alpha."""
    stock_returns = pd.DataFrame(stock_returns).astype(float)
    factor_returns = pd.DataFrame(factor_returns).astype(float)
    beta = pd.DataFrame(beta).set_index("Stock")
    beta = beta.loc[stock_returns.columns, factor_returns.columns].to_numpy(dtype=float)
    weights = np.asarray(weights, dtype=float).copy()

    daily_contributions = []
    portfolio_returns = []
    for day in range(len(stock_returns)):
        stock_r = stock_returns.iloc[day].to_numpy()
        factor_r = factor_returns.iloc[day].to_numpy()

        # Factor contribution = current factor exposure times today's factor return.
        factor_exposure = weights @ beta
        factor_contributions = factor_exposure * factor_r
        unexplained = weights @ (stock_r - beta @ factor_r)
        daily_contributions.append(np.r_[factor_contributions, unexplained])

        portfolio_return = weights @ stock_r
        portfolio_returns.append(portfolio_return)
        weights = weights * (1 + stock_r) / (1 + portfolio_return)

    daily_contributions = np.array(daily_contributions)
    portfolio_returns = np.array(portfolio_returns)
    total_return = np.prod(1 + portfolio_returns) - 1
    linking = np.ones(len(portfolio_returns))
    valid = np.abs(portfolio_returns) > 1e-12
    linking[valid] = np.log1p(portfolio_returns[valid]) / portfolio_returns[valid]
    linking /= np.log1p(total_return) / total_return

    return_contributions = (daily_contributions * linking[:, None]).sum(axis=0)
    portfolio_sd = np.std(portfolio_returns, ddof=1)
    volatility_contributions = (
        np.cov(daily_contributions.T, portfolio_returns, ddof=1)[:-1, -1]
        / portfolio_sd
    )
    factor_total_returns = np.prod(1 + factor_returns.to_numpy(), axis=0) - 1
    alpha_total_return = np.prod(1 + daily_contributions[:, -1]) - 1
    total_returns = np.r_[factor_total_returns, alpha_total_return]

    result = pd.DataFrame(
        [total_returns, return_contributions, volatility_contributions],
        index=["TotalReturn", "Return Attribution", "Vol Attribution"],
        columns=list(factor_returns.columns) + ["Alpha"],
    )
    result.insert(0, "Value", result.index)
    result["Portfolio"] = [total_return, return_contributions.sum(), portfolio_sd]
    return result.reset_index(drop=True)

# =========================================================
# 12. European and American options
# =========================================================

def gbsm_greeks(option_type, stock, strike, maturity, rate, dividend_yield, volatility):
    """Generalized BSM price and Greeks for a European call or put.

    Rates are continuously compounded. Theta is calendar-time decay, i.e.
    minus the derivative with respect to remaining maturity.
    """
    carry = rate - dividend_yield
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(stock / strike) + (carry + 0.5 * volatility**2) * maturity) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount_stock = np.exp((carry - rate) * maturity)
    discount_strike = np.exp(-rate * maturity)

    if option_type.lower() == "call":
        value = stock * discount_stock * stats.norm.cdf(d1) - strike * discount_strike * stats.norm.cdf(d2)
        delta = discount_stock * stats.norm.cdf(d1)
        rho = strike * maturity * discount_strike * stats.norm.cdf(d2)
        theta = (-stock * discount_stock * stats.norm.pdf(d1) * volatility / (2 * sqrt_t)
                 - (carry - rate) * stock * discount_stock * stats.norm.cdf(d1)
                 - rate * strike * discount_strike * stats.norm.cdf(d2))
    else:
        value = strike * discount_strike * stats.norm.cdf(-d2) - stock * discount_stock * stats.norm.cdf(-d1)
        delta = discount_stock * (stats.norm.cdf(d1) - 1)
        rho = -strike * maturity * discount_strike * stats.norm.cdf(-d2)
        theta = (-stock * discount_stock * stats.norm.pdf(d1) * volatility / (2 * sqrt_t)
                 + (carry - rate) * stock * discount_stock * stats.norm.cdf(-d1)
                 + rate * strike * discount_strike * stats.norm.cdf(-d2))

    gamma = discount_stock * stats.norm.pdf(d1) / (stock * volatility * sqrt_t)
    vega = stock * discount_stock * stats.norm.pdf(d1) * sqrt_t
    return np.array([value, delta, gamma, vega, rho, theta])


def american_option_tree(option_type, stock, strike, maturity, rate, carry, volatility, steps=500):
    """CRR recombining tree for American options with continuous dividend yield."""
    dt = maturity / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1 / up
    probability = (np.exp(carry * dt) - down) / (up - down)
    discount = np.exp(-rate * dt)

    j = np.arange(steps + 1)
    stock_prices = stock * up**j * down**(steps - j)
    if option_type.lower() == "call":
        values = np.maximum(stock_prices - strike, 0)
    else:
        values = np.maximum(strike - stock_prices, 0)

    for step in range(steps - 1, -1, -1):
        values = discount * (probability * values[1:] + (1 - probability) * values[:-1])
        j = np.arange(step + 1)
        stock_prices = stock * up**j * down**(step - j)
        if option_type.lower() == "call":
            exercise = np.maximum(stock_prices - strike, 0)
        else:
            exercise = np.maximum(strike - stock_prices, 0)
        values = np.maximum(values, exercise)

    return values[0]


def american_option_greeks(option_type, stock, strike, maturity, rate, dividend_yield, volatility, steps=500):
    """American tree Greeks using centered bumps.

    The course test uses a 1.5 stock bump for gamma (to smooth the tree's
    piecewise-linear payoff) and a much smaller bump for delta. Its rho
    holds cost of carry b=r-q fixed, and its theta is dV/dT, rather than
    the usual calendar-time theta sign for the European GBSM result.
    """
    carry = rate - dividend_yield

    def price(s=stock, t=maturity, r=rate, vol=volatility):
        return american_option_tree(option_type, s, strike, t, r, carry, vol, steps)

    value = price()
    h_delta = 0.001
    h_gamma = 1.5
    h_vega = 1e-5
    h_rate = 5e-6
    h_time = 1e-4

    delta = (price(s=stock + h_delta) - price(s=stock - h_delta)) / (2 * h_delta)
    gamma = (price(s=stock + h_gamma) + price(s=stock - h_gamma) - 2 * value) / h_gamma**2
    vega = (price(vol=volatility + h_vega) - price(vol=volatility - h_vega)) / (2 * h_vega)
    rho = (price(r=rate + h_rate) - price(r=rate - h_rate)) / (2 * h_rate)
    theta = (price(t=maturity + h_time) - price(t=maturity - h_time)) / (2 * h_time)

    return np.array([value, delta, gamma, vega, rho, theta])


def _american_tree_for_many_stocks(prices, strike, maturity, rate, volatility, option_type, steps):
    """Price the same no-dividend American option for many starting stock prices."""
    prices = np.asarray(prices, dtype=float)
    dt = maturity / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1 / up
    probability = (np.exp(rate * dt) - down) / (up - down)
    discount = np.exp(-rate * dt)

    nodes = np.arange(steps + 1)
    stock_nodes = prices[:, None] * up**nodes * down**(steps - nodes)
    is_call = option_type.lower() == "call"
    values = np.maximum(stock_nodes - strike, 0) if is_call else np.maximum(strike - stock_nodes, 0)

    for step in range(steps - 1, -1, -1):
        values = discount * (probability * values[:, 1:] + (1 - probability) * values[:, :-1])
        nodes = np.arange(step + 1)
        stock_nodes = prices[:, None] * up**nodes * down**(step - nodes)
        exercise = np.maximum(stock_nodes - strike, 0) if is_call else np.maximum(strike - stock_nodes, 0)
        values = np.maximum(values, exercise)
    return values[:, 0]


def american_option_discrete_dividends(option_type, stock, strike, maturity, rate,
                                       volatility, dividend_dates, dividend_amounts,
                                       steps=500, grid_size=40000):
    """American option with two fixed cash dividends using separate tree segments.

    Each dividend starts a separate subtree (the tree does not recombine
    across a fixed cash payment). A dense grid/interpolation at the final
    dividend avoids rebuilding many identical subtrees individually.
    Dividend dates are expressed in days, on the same 365-day basis as T.
    """
    if len(dividend_dates) != 2 or len(dividend_amounts) != 2:
        raise ValueError("This test function expects two discrete dividends")

    dt = maturity / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1 / up
    probability = (np.exp(rate * dt) - down) / (up - down)
    discount = np.exp(-rate * dt)
    is_call = option_type.lower() == "call"

    first_step, second_step = [int(round(day * steps / (maturity * 365))) for day in dividend_dates]
    first_dividend, second_dividend = dividend_amounts

    # Stock prices immediately BEFORE the first cash dividend.
    j = np.arange(first_step + 1)
    first_prices = stock * up**j * down**(first_step - j)
    after_first = first_prices - first_dividend

    # Each post-dividend stock price now creates its own recombining subtree.
    middle_steps = second_step - first_step
    j = np.arange(middle_steps + 1)
    second_prices = after_first[:, None] * up**j * down**(middle_steps - j)

    # Value the remaining part after the second dividend on a dense stock grid.
    low = max(0.001, float(np.min(second_prices)) - second_dividend)
    high = float(np.max(second_prices))
    grid = np.linspace(np.log(low), np.log(high), grid_size)
    grid_prices = np.exp(grid)
    post_dividend_values = _american_tree_for_many_stocks(
        grid_prices, strike, (steps - second_step) * dt, rate,
        volatility, option_type, steps - second_step
    )
    continuation = np.interp(
        np.log(np.maximum(second_prices - second_dividend, 1e-9)),
        grid, post_dividend_values
    )
    exercise = np.maximum(second_prices - strike, 0) if is_call else np.maximum(strike - second_prices, 0)
    values = np.maximum(continuation, exercise)

    # Backward induction from dividend 2 to dividend 1.
    for step in range(middle_steps - 1, -1, -1):
        values = discount * (probability * values[:, 1:] + (1 - probability) * values[:, :-1])
        j = np.arange(step + 1)
        prices = after_first[:, None] * up**j * down**(step - j)
        exercise = np.maximum(prices - strike, 0) if is_call else np.maximum(strike - prices, 0)
        values = np.maximum(values, exercise)

    # At the first dividend, compare exercise BEFORE payment with holding.
    exercise = np.maximum(first_prices - strike, 0) if is_call else np.maximum(strike - first_prices, 0)
    values = np.maximum(values[:, 0], exercise)

    # The first segment is a standard recombining American tree.
    for step in range(first_step - 1, -1, -1):
        values = discount * (probability * values[1:] + (1 - probability) * values[:-1])
        j = np.arange(step + 1)
        prices = stock * up**j * down**(step - j)
        exercise = np.maximum(prices - strike, 0) if is_call else np.maximum(strike - prices, 0)
        values = np.maximum(values, exercise)

    return values[0]

# =========================================================
# 13. Multivariate t and copula models
# =========================================================

def kendall_correlation(data):
    """Convert pairwise Kendall's tau to Gaussian/t copula correlations."""
    data = pd.DataFrame(data)
    n = data.shape[1]
    correlation = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau = stats.kendalltau(data.iloc[:, i], data.iloc[:, j]).statistic
            correlation[i, j] = np.sin(np.pi * tau / 2)
            correlation[j, i] = correlation[i, j]

    # The pairwise tau estimate may not be positive definite.
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if eigenvalues.min() <= 0:
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(repaired))
        correlation = repaired / np.outer(scale, scale)
    return pd.DataFrame(correlation, index=data.columns, columns=data.columns)


def fit_multivariate_t(data, correlation=None):
    """Profile t degrees of freedom using Kendall correlation and sample SDs.

    The fitted center is the sample mean; covariance is made from the sample
    standard deviations and Kendall correlations. t scale = cov*(nu-2)/nu.
    """
    from scipy.optimize import minimize_scalar
    x = pd.DataFrame(data).to_numpy(dtype=float)
    center = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    if correlation is None:
        correlation = kendall_correlation(data)
    covariance = np.asarray(correlation, dtype=float) * np.outer(sd, sd)

    def log_likelihood(nu):
        scale = covariance * (nu - 2) / nu
        return stats.multivariate_t.logpdf(x, loc=center, shape=scale, df=nu).sum()

    result = minimize_scalar(lambda nu: -log_likelihood(nu),
                             bounds=(2.01, 100), method="bounded",
                             options={"xatol": 1e-10})
    nu = float(result.x)
    scale = covariance * (nu - 2) / nu
    return center, scale, nu, float(-result.fun)


def fitted_t_uniforms(data):
    """Fit a Student-t marginal to each asset and transform returns to U(0,1)."""
    x = pd.DataFrame(data)
    u = np.zeros(x.shape)
    fits = []
    for i, stock in enumerate(x.columns):
        nu, mu, sigma = stats.t.fit(x[stock].to_numpy())
        u[:, i] = stats.t.cdf(x[stock], df=nu, loc=mu, scale=sigma)
        fits.append((nu, mu, sigma))
    return np.clip(u, 1e-12, 1-1e-12), fits


def gaussian_copula_loglik(uniforms, correlation):
    """Gaussian joint log density minus the standard normal marginal densities."""
    z = stats.norm.ppf(uniforms)
    joint = stats.multivariate_normal.logpdf(z, cov=np.asarray(correlation))
    marginal = stats.norm.logpdf(z).sum(axis=1)
    return float(np.sum(joint - marginal))


def t_copula_loglik(uniforms, correlation, nu):
    """Student-t copula log likelihood for a fixed correlation and df."""
    q = stats.t.ppf(uniforms, df=nu)
    joint = stats.multivariate_t.logpdf(q, shape=np.asarray(correlation), df=nu)
    marginal = stats.t.logpdf(q, df=nu).sum(axis=1)
    return float(np.sum(joint - marginal))


def fit_t_copula(uniforms, correlation):
    """Profile the single free copula parameter: degrees of freedom."""
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(
        lambda nu: -t_copula_loglik(uniforms, correlation, nu),
        bounds=(2.01, 100), method="bounded", options={"xatol": 1e-10}
    )
    return float(result.x), float(-result.fun)


def copula_information_criteria(ll, k, n):
    """AICc and BIC with k free copula parameters, not marginal parameters."""
    aicc = 2*k - 2*ll + 2*k*(k+1)/(n-k-1)
    bic = np.log(n)*k - 2*ll
    return aicc, bic


def t_copula_tail_dependence(correlation, nu):
    """Return a table of bivariate lower-tail dependence coefficients."""
    correlation = np.asarray(correlation, dtype=float)
    rows = []
    for i in range(correlation.shape[0]):
        for j in range(i + 1, correlation.shape[0]):
            rho = correlation[i, j]
            z = -np.sqrt((nu+1)*(1-rho)/(1+rho))
            tail = 2*stats.t.cdf(z, df=nu+1)
            rows.append([i+1, j+1, rho, tail])
    return pd.DataFrame(rows, columns=["I", "J", "Rho", "Lambda"])


def t_copula_portfolio_risk(returns, portfolio, n_sim=300000, seed=1234, alpha=0.05):
    """Simulate five fitted t marginals joined by a fitted Student-t copula."""
    returns = pd.DataFrame(returns)
    portfolio = pd.DataFrame(portfolio)
    correlation = kendall_correlation(returns)
    uniforms, fits = fitted_t_uniforms(returns)
    nu, _ = fit_t_copula(uniforms, correlation)

    rng = np.random.default_rng(seed)
    # Multivariate t: correlated normals divided by a common chi-square scale.
    z = rng.multivariate_normal(np.zeros(len(fits)), correlation, size=n_sim)
    chi = rng.chisquare(nu, size=n_sim)
    simulated_t = z / np.sqrt(chi[:, None] / nu)
    simulated_uniforms = stats.t.cdf(simulated_t, df=nu)

    simulated_returns = np.zeros_like(simulated_uniforms)
    for i, (df, mu, sigma) in enumerate(fits):
        simulated_returns[:, i] = stats.t.ppf(simulated_uniforms[:, i], df, loc=mu, scale=sigma)

    results = []
    total_pnl = np.zeros(n_sim)
    total_value = 0.0
    for i, position in portfolio.reset_index(drop=True).iterrows():
        value = float(position["currentValue"])
        pnl = value * simulated_returns[:, i]
        total_pnl += pnl
        total_value += value
        cutoff = np.quantile(pnl, alpha)
        var = -cutoff
        es = -pnl[pnl <= cutoff].mean()
        results.append([position["Stock"], var, es, var/value, es/value])

    cutoff = np.quantile(total_pnl, alpha)
    var = -cutoff
    es = -total_pnl[total_pnl <= cutoff].mean()
    results.append(["Total", var, es, var/total_value, es/total_value])
    return pd.DataFrame(results, columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
