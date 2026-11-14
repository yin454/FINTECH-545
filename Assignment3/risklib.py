import numpy as np
from scipy import stats
from scipy.optimize import brentq


# =============================================================================
# Option pricing: Generalized Black-Scholes-Merton
# =============================================================================
def gbs_price(S, K, T, r, b, sigma, option_type):
    """European price; b is cost of carry (b = r - dividend yield)."""
    if T <= 0:
        if option_type == "Call":
            return np.maximum(np.asarray(S) - K, 0)
        return np.maximum(K - np.asarray(S), 0)

    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discounted_stock = S * np.exp((b - r) * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == "Call":
        return discounted_stock * stats.norm.cdf(d1) - discounted_strike * stats.norm.cdf(d2)
    return discounted_strike * stats.norm.cdf(-d2) - discounted_stock * stats.norm.cdf(-d1)


def option_greeks(S, K, T, r, b, sigma, option_type):
    """Return analytic delta, gamma, annual theta and vega of a European option."""
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    carry_discount = np.exp((b - r) * T)
    pdf_d1 = stats.norm.pdf(d1)

    vega = S * carry_discount * pdf_d1 * np.sqrt(T)
    gamma = carry_discount * pdf_d1 / (S * sigma * np.sqrt(T))
    theta = -S * carry_discount * pdf_d1 * sigma / (2 * np.sqrt(T))

    if option_type == "Call":
        delta = carry_discount * stats.norm.cdf(d1)
        theta -= ((b - r) * S * carry_discount * stats.norm.cdf(d1)
                  + r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        delta = carry_discount * (stats.norm.cdf(d1) - 1)
        theta += ((b - r) * S * carry_discount * stats.norm.cdf(-d1)
                  + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def implied_volatility(price, S, K, T, r, b, option_type):
    """Find volatility matching price; return NaN if no valid IV exists."""
    minimum = float(gbs_price(S, K, T, r, b, 1e-7, option_type))
    if option_type == "Call":
        maximum = float(S * np.exp((b - r) * T))
    else:
        maximum = float(K * np.exp(-r * T))

    if price < minimum - 1e-8 or price > maximum + 1e-8:
        return np.nan
    if abs(price - minimum) < 1e-9:
        return 0.0

    def pricing_error(volatility):
        return float(gbs_price(S, K, T, r, b, volatility, option_type)) - price

    try:
        return brentq(pricing_error, 1e-6, 10, xtol=1e-12)
    except ValueError:
        return np.nan


# =============================================================================
# Binomial tree for European and American options
# =============================================================================
def binomial_tree(S, K, T, r, sigma, N, option_type, american=False, dividends=()):
    """CRR tree with fixed cash dividends at dates aligned to tree steps.

    Each dividend node starts a new recombining subtree. Across the cash
    dividend date, subtrees are not recombined. American options may be
    exercised at any node, including just before a dividend is paid.
    """
    dt = T / N
    up = np.exp(sigma * np.sqrt(dt))
    down = 1 / up
    probability = (np.exp(r * dt) - down) / (up - down)
    discount = np.exp(-r * dt)
    if not (0 <= probability <= 1):
        raise ValueError("The CRR probability must be between 0 and 1.")

    events = [(int(round(date / dt)), float(amount))
              for date, amount in dividends if amount != 0]
    events.sort()
    if any(abs(step * dt - date) > 1e-8
           for (date, _), (step, _) in zip(dividends, events)):
        raise ValueError("Dividend dates must fall exactly on tree nodes.")

    def payoff(stock):
        if option_type == "Call":
            return np.maximum(stock - K, 0)
        return np.maximum(K - stock, 0)

    def price_segment(starting_prices, steps, next_segment=None, dividend=0):
        """Backward induction within one recombining section of the tree."""
        start = np.asarray(starting_prices, dtype=float)
        up_moves = np.arange(steps + 1)
        terminal_stock = (start[..., None] * up**up_moves
                          * down**(steps - up_moves))

        if next_segment is None:
            option_values = payoff(terminal_stock)
        else:
            # At a dividend date, continuation uses the ex-dividend price.
            after_dividend = np.maximum(terminal_stock - dividend, 1e-10)
            option_values = next_segment(after_dividend)
            if american:
                # Compare exercising immediately before dividend vs holding.
                option_values = np.maximum(option_values, payoff(terminal_stock))

        for step in range(steps - 1, -1, -1):
            option_values = discount * (
                probability * option_values[..., 1:step + 2]
                + (1 - probability) * option_values[..., :step + 1]
            )
            if american:
                up_moves = np.arange(step + 1)
                node_stock = (start[..., None] * up**up_moves
                              * down**(step - up_moves))
                option_values = np.maximum(option_values, payoff(node_stock))

        return option_values[..., 0]

    event_dates = [(0, 0.0)] + events + [(N, 0.0)]

    def price_from_event(event_index, starting_prices):
        steps = event_dates[event_index + 1][0] - event_dates[event_index][0]
        if event_index == len(event_dates) - 2:
            return price_segment(starting_prices, steps)

        next_dividend = event_dates[event_index + 1][1]
        return price_segment(
            starting_prices, steps,
            next_segment=lambda stocks: price_from_event(event_index + 1, stocks),
            dividend=next_dividend,
        )

    return float(price_from_event(0, np.array(S)))


# =============================================================================
# Return statistics and distribution fitting
# =============================================================================
def sample_moments(returns):
    """Mean, sample SD, bias-corrected skew and excess kurtosis."""
    x = np.asarray(returns, dtype=float)
    return {
        "mean": np.mean(x),
        "stdev": np.std(x, ddof=1),
        "skew": stats.skew(x, bias=False),
        "excess_kurtosis": stats.kurtosis(x, bias=False),
    }


def aicc(log_likelihood, parameters, observations):
    """Small-sample corrected Akaike information criterion."""
    k, n = parameters, observations
    return 2 * k - 2 * log_likelihood + 2 * k * (k + 1) / (n - k - 1)


def fit_return_distributions(returns):
    """Fit Normal and Student t by MLE and report their AICc."""
    x = np.asarray(returns, dtype=float)
    normal_mean, normal_sd = stats.norm.fit(x)
    t_df, t_location, t_scale = stats.t.fit(x)
    normal_ll = np.sum(stats.norm.logpdf(x, normal_mean, normal_sd))
    t_ll = np.sum(stats.t.logpdf(x, t_df, t_location, t_scale))
    return {
        "normal": [normal_mean, normal_sd],
        "student_t": [t_df, t_location, t_scale],
        "aicc_normal": aicc(normal_ll, 2, len(x)),
        "aicc_t": aicc(t_ll, 3, len(x)),
    }


# =============================================================================
# VaR and ES: positive numbers represent losses
# =============================================================================
def pnl_var_es(pnl):
    """Compute mean P&L and historical VaR/ES at 5% and 1%."""
    pnl = np.asarray(pnl, dtype=float)
    loss = -pnl
    output = {"MeanPnL": np.mean(pnl)}
    for alpha in [0.05, 0.01]:
        cutoff = np.quantile(loss, 1 - alpha)
        output[f"VaR{int(alpha * 100)}"] = cutoff
        output[f"ES{int(alpha * 100)}"] = np.mean(loss[loss >= cutoff])
    return output


def portfolio_risk_statistics(returns):
    """Historical and Normal VaR/ES for a monthly portfolio return series."""
    x = np.asarray(returns, dtype=float)
    mean = x.mean()
    sd = x.std(ddof=1)
    loss = -x
    output = {}

    for alpha in [0.05, 0.01]:
        cutoff = np.quantile(loss, 1 - alpha)
        z = stats.norm.ppf(1 - alpha)
        suffix = str(int(alpha * 100))
        output["HistVaR" + suffix] = cutoff
        output["HistES" + suffix] = loss[loss >= cutoff].mean()
        output["NormVaR" + suffix] = -mean + z * sd
        output["NormES" + suffix] = -mean + sd * stats.norm.pdf(z) / alpha
    return output
