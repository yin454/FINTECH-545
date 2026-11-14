from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq, minimize
import statsmodels.api as sm

import risklib as rl

BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

# Fixed random seeds make the Monte Carlo answers reproducible.
RESULTS = {}


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def money(value):
    return f"${value:,.2f}"


def save_results():
    """Save the numbers in one file, in addition to printing them."""
    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.replace({np.nan: None}).to_dict("records")
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    output = convert(RESULTS)
    (BASE / "results.json").write_text(
        json.dumps(output, indent=2, allow_nan=True, default=str), encoding="utf-8"
    )


# =============================================================================
# Problem 1 - Implied volatility from a dirty option chain
# =============================================================================
section("Problem 1 - Implied Volatility")

p1 = pd.read_csv(BASE / "problem1.csv")
stock_price = 182.40
risk_free = 0.0425
dividend_yield = 0.0060
T = 35 / 365  # Calendar days from Nov 13 to Dec 18, 2026
b = risk_free - dividend_yield  # Cost of carry in generalized BSM

# Calculate the mid price and bid/ask spread for each quote.
p1["Mid"] = (p1["Bid"] + p1["Ask"]) / 2
p1["Spread"] = p1["Ask"] - p1["Bid"]

# Invert the option pricing formula at the bid, mid and ask prices.
for quote in ["Mid", "Bid", "Ask"]:
    p1["IV_" + quote] = [
        rl.implied_volatility(row[quote], stock_price, row.Strike, T,
                              risk_free, b, row.Type)
        for _, row in p1.iterrows()
    ]

# Vega measures how much the option price changes per 1.00 change in volatility.
p1["Vega"] = [
    rl.option_greeks(stock_price, row.Strike, T, risk_free,
                     b, row.IV_Mid, row.Type)["vega"]
    if np.isfinite(row.IV_Mid) and row.IV_Mid > 0 else np.nan
    for _, row in p1.iterrows()
]
p1["IV_per_cent"] = 0.01 / p1["Vega"]
p1.to_csv(BASE / "problem1_quotes.csv", index=False)

# Check put-call parity separately at every strike.
calls = p1[p1.Type == "Call"].set_index("Strike")
puts = p1[p1.Type == "Put"].set_index("Strike")
strikes = calls.index.to_numpy()
parity_value = (stock_price * np.exp(-dividend_yield * T)
                - strikes * np.exp(-risk_free * T))
parity_gap = (calls.Mid - puts.Mid).to_numpy() - parity_value
spread_band = (calls.Spread + puts.Spread).to_numpy()

parity_table = pd.DataFrame({
    "Strike": strikes,
    "ParityGap": parity_gap,
    "SpreadBand": spread_band,
    "Flag": np.abs(parity_gap) > spread_band,
    "CallIV": calls.IV_Mid.to_numpy(),
    "PutIV": puts.IV_Mid.to_numpy(),
    "CallVega": calls.Vega.to_numpy(),
    "PutVega": puts.Vega.to_numpy(),
})
parity_table.to_csv(BASE / "problem1_parity.csv", index=False)
print("Put-call parity gaps and spread bands:\n", parity_table.to_string(index=False))
print("\nQuote IVs, vegas, and IV change for a one-cent price move:\n",
      p1[["Type", "Strike", "IV_Bid", "IV_Mid", "IV_Ask", "Vega", "IV_per_cent"]].to_string(index=False))

plt.figure(figsize=(8, 5))
for option_type, marker in [("Call", "o"), ("Put", "s")]:
    part = p1[p1.Type == option_type]
    plt.plot(part.Strike, part.IV_Mid * 100, marker=marker,
             label=option_type + " mid IV")
    plt.fill_between(part.Strike,
                     (part.IV_Bid * 100).to_numpy(),
                     (part.IV_Ask * 100).to_numpy(), alpha=0.15)
plt.title("Problem 1: Implied Volatility by Strike")
plt.xlabel("Strike ($)")
plt.ylabel("Implied volatility (%)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "p1_smile.png", dpi=150)
plt.close()
RESULTS["p1"] = {"parity": parity_table, "quotes": p1}
save_results()


# =============================================================================
# Problem 2 - American options with discrete dividends
# =============================================================================
section("Problem 2 - American Options and Discrete Dividends")

S = 48.50
K = 50.00
T = 0.50
r = 0.045
vol = 0.30
dividends = [(0.10, 0.25), (0.40, 1.50)]
N = 100  # Multiple of 5 so that both dividend dates are tree nodes

# Week 7 condition for early exercise just before a dividend.
for date, amount in dividends:
    threshold = K * (1 - np.exp(-r * (T - date)))
    print(f"Dividend {amount:.2f} at t={date:.2f}: threshold={threshold:.4f}, "
          f"eligible={amount > threshold}")

no_dividend = {}
with_dividend = {}
pv_approximation = {}
present_value_dividends = sum(amount * np.exp(-r * date)
                              for date, amount in dividends)

for option_type in ["Call", "Put"]:
    # No dividends: GBSM for Europeans and a CRR tree for Americans.
    european = rl.gbs_price(S, K, T, r, r, vol, option_type)
    american = (european if option_type == "Call" else
                rl.binomial_tree(S, K, T, r, vol, 400, option_type, True))
    no_dividend["European" + option_type] = european
    no_dividend["American" + option_type] = american

    # Cash dividends: use the non-recombining binomial tree.
    with_dividend["European" + option_type] = rl.binomial_tree(
        S, K, T, r, vol, N, option_type, False, dividends
    )
    with_dividend["American" + option_type] = rl.binomial_tree(
        S, K, T, r, vol, N, option_type, True, dividends
    )

    # Simpler approximation for the European option: subtract dividend PV.
    pv_approximation["European" + option_type] = rl.gbs_price(
        S - present_value_dividends, K, T, r, r, vol, option_type
    )

print("\nPrices without dividends:", no_dividend)
print("Prices with dividends:", with_dividend)
print("European GBSM PV-dividend approximations:", pv_approximation)

# Part (d): neighboring N values illustrate binomial discretization effects.
convergence = []
for steps in [25, 50, 75, 100, 105, 125, 150, 200]:
    row = {"N": steps}
    for option_type in ["Call", "Put"]:
        row["American" + option_type] = rl.binomial_tree(
            S, K, T, r, vol, steps, option_type, True, dividends
        )
    convergence.append(row)
convergence_table = pd.DataFrame(convergence)
print("\nAmerican option convergence:\n", convergence_table.to_string(index=False))

plt.figure(figsize=(7, 4))
for option_type in ["Call", "Put"]:
    plt.plot(convergence_table.N, convergence_table["American" + option_type],
             "-o", label="American " + option_type)
plt.title("Problem 2: Tree Convergence")
plt.xlabel("Number of steps")
plt.ylabel("Option price ($)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "p2_convergence.png", dpi=150)
plt.close()

# Part (e): central finite differences for Delta and Gamma.
american_greeks = {}
for option_type in ["Call", "Put"]:
    american_greeks[option_type] = {}
    base_price = with_dividend["American" + option_type]
    for bump in [1, 0.1, 0.01, 0.001]:
        price_up = rl.binomial_tree(S + bump, K, T, r, vol, N,
                                    option_type, True, dividends)
        price_down = rl.binomial_tree(S - bump, K, T, r, vol, N,
                                      option_type, True, dividends)
        delta = (price_up - price_down) / (2 * bump)
        gamma = (price_up - 2 * base_price + price_down) / bump**2
        american_greeks[option_type][str(bump)] = {"Delta": delta, "Gamma": gamma}
    print(f"\n{option_type} finite-difference Greeks:", american_greeks[option_type])

# Part (f): bump one cash dividend at a time and reprice.
dividend_sensitivity = {}
dividend_bump = 0.0001
for option_type in ["Call", "Put"]:
    dividend_sensitivity[option_type] = {}
    for j in range(2):
        dividends_up = dividends.copy()
        dividends_down = dividends.copy()
        date, amount = dividends[j]
        dividends_up[j] = (date, amount + dividend_bump)
        dividends_down[j] = (date, amount - dividend_bump)

        sensitivity = {}
        for style, is_american in [("European", False), ("American", True)]:
            up_price = rl.binomial_tree(S, K, T, r, vol, N,
                                        option_type, is_american, dividends_up)
            down_price = rl.binomial_tree(S, K, T, r, vol, N,
                                          option_type, is_american, dividends_down)
            sensitivity[style] = (up_price - down_price) / (2 * dividend_bump)
        dividend_sensitivity[option_type]["Dividend" + str(j + 1)] = sensitivity
print("\nSensitivity to each dividend:", dividend_sensitivity)

# The early exercise premium is American price minus European price.
early_premia = {}
for label, prices in [("NoDiv", no_dividend), ("WithDiv", with_dividend)]:
    early_premia[label] = {}
    for option_type in ["Call", "Put"]:
        early_premia[label][option_type] = (prices["American" + option_type]
                                              - prices["European" + option_type])
print("\nEarly exercise premia:", early_premia)

# Critical stock price just before the second dividend. The next 0.1 year
# is a 20-step tree and the stock price just after the dividend is S-1.50.
def exercise_gap(stock):
    immediate = stock - K
    continuation = rl.binomial_tree(stock - 1.50, K, 0.10, r, vol,
                                     20, "Call", True)
    return immediate - continuation

stock_grid = np.linspace(K + 0.00001, 200, 200)
critical_prices = []
for left, right in zip(stock_grid[:-1], stock_grid[1:]):
    if exercise_gap(left) * exercise_gap(right) < 0:
        critical_prices.append(brentq(exercise_gap, left, right))
print("Critical stock price before second dividend:", critical_prices)

RESULTS["p2"] = {
    "conditions": [
        {"dividend": amount, "date": date,
         "threshold": K * (1 - np.exp(-r * (T - date))),
         "eligible": bool(amount > K * (1 - np.exp(-r * (T - date))))}
        for date, amount in dividends
    ],
    "no_dividend": no_dividend, "with_dividend": with_dividend,
    "pv_dividend_gbsm": pv_approximation, "convergence": convergence,
    "greeks": american_greeks, "dividend_sensitivity": dividend_sensitivity,
    "early_premia": early_premia, "exercise_threshold": critical_prices,
}
save_results()


# =============================================================================
# Problem 3 - 10-day VaR for a book of options
# =============================================================================
section("Problem 3 - VaR of an Option Book")

p3_prices = pd.read_csv(BASE / "problem3.csv")
p3_positions = pd.read_csv(BASE / "problem3_portfolios.csv")
p3_returns = p3_prices["ABC"].pct_change().dropna().to_numpy()
S = float(p3_prices.ABC.iloc[-1])
r = 0.0425
trading_days = 255
horizon = 10

# Fit the daily returns first, then compare the Normal and t AICc values.
return_moments = rl.sample_moments(p3_returns)
distribution_fit = rl.fit_return_distributions(p3_returns)
print("First four moments:", return_moments)
print("Normal and Student t fits:", distribution_fit)

# Value a whole book using its holdings. The same function works for one
# price or for a vector of simulated prices. Days to expiry also change.
def book_value(book_name, underlying_price, days_elapsed=0):
    prices = np.asarray(underlying_price, dtype=float)
    total = np.zeros_like(prices, dtype=float)
    book = p3_positions[p3_positions.Portfolio == book_name]

    for _, position in book.iterrows():
        holding = position.Holding
        if position.Type == "Stock":
            total += holding * prices
        else:
            remaining = (position.DaysToExpiry - days_elapsed) / trading_days
            total += holding * rl.gbs_price(
                prices, position.Strike, remaining, r, r,
                position.ImpliedVol, position.Type
            )
    return total

# Add up each position's delta, gamma and theta in each portfolio.
book_greeks = {}
initial_values = {}
for book_name, book in p3_positions.groupby("Portfolio", sort=False):
    initial_values[book_name] = float(book_value(book_name, S))
    delta = 0.0
    gamma = 0.0
    theta = 0.0

    for _, position in book.iterrows():
        holding = position.Holding
        if position.Type == "Stock":
            delta += holding
        else:
            T_option = position.DaysToExpiry / trading_days
            g = rl.option_greeks(S, position.Strike, T_option, r, r,
                                 position.ImpliedVol, position.Type)
            delta += holding * g["delta"]
            gamma += holding * g["gamma"]
            theta += holding * g["theta"]

    book_greeks[book_name] = {
        "Value": initial_values[book_name], "Delta": delta, "Gamma": gamma,
        "Theta_annual": theta, "Theta_10d": theta * horizon / trading_days,
    }
print("\nPortfolio Greeks:\n", pd.DataFrame(book_greeks).T)

# Simulate 10 daily arithmetic returns and compound them into final prices.
rng = np.random.default_rng(54503)
n_simulations = 200_000
normal_sigma = distribution_fit["normal"][1]
t_df, t_location, t_scale = distribution_fit["student_t"]

normal_returns = rng.normal(0, normal_sigma, (n_simulations, horizon))
t_returns = stats.t.rvs(t_df, loc=0, scale=t_scale,
                         size=(n_simulations, horizon), random_state=rng)
simulated_returns = {"Normal": normal_returns, "StudentT": t_returns}

simulation_results = {}
gamma_approximations = {}
for book_name in book_greeks:
    simulation_results[book_name] = {}
    gamma_approximations[book_name] = {}
    greek = book_greeks[book_name]

    for model_name, daily_returns in simulated_returns.items():
        final_price = S * np.prod(1 + daily_returns, axis=1)
        price_change = final_price - S

        # Full revaluation: subtract original book value from future value.
        pnl_full = book_value(book_name, final_price, horizon) - initial_values[book_name]
        simulation_results[book_name][model_name] = rl.pnl_var_es(pnl_full)

        # Second-order Taylor approximation: delta + gamma + theta.
        pnl_approx = (greek["Delta"] * price_change
                      + 0.5 * greek["Gamma"] * price_change**2
                      + greek["Theta_10d"])
        gamma_approximations[book_name][model_name] = rl.pnl_var_es(pnl_approx)

# Delta-normal assumes a linear exposure and normally distributed P&L.
delta_normal = {}
for book_name, greek in book_greeks.items():
    ten_day_sd = abs(greek["Delta"]) * S * normal_sigma * np.sqrt(horizon)
    delta_normal[book_name] = {}
    for alpha in [0.05, 0.01]:
        z = stats.norm.ppf(1 - alpha)
        delta_normal[book_name][f"VaR{int(alpha * 100)}"] = z * ten_day_sd
        delta_normal[book_name][f"ES{int(alpha * 100)}"] = (
            stats.norm.pdf(z) * ten_day_sd / alpha
        )

    # Compare the true P&L curve with the straight delta-only line.
    price_grid = np.linspace(S * 0.55, S * 1.45, 451)
    pnl_curve = book_value(book_name, price_grid, horizon) - initial_values[book_name]
    linear_curve = greek["Delta"] * (price_grid - S)
    plt.figure(figsize=(7, 4))
    plt.plot(price_grid, pnl_curve, label="Full revaluation")
    plt.plot(price_grid, linear_curve, "--", label="Delta only")
    plt.axhline(0, linewidth=0.7, color="gray")
    plt.title(book_name)
    plt.xlabel("ABC price after 10 days ($)")
    plt.ylabel("P&L ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"p3_{book_name}.png", dpi=140)
    plt.close()

print("\nDelta-normal VaR/ES:\n", pd.DataFrame(delta_normal).T)
for name in book_greeks:
    print(f"\n{name} simulated P&L / VaR / ES:")
    print(pd.DataFrame(simulation_results[name]).T)
    print("Delta-gamma-theta approximation:")
    print(pd.DataFrame(gamma_approximations[name]).T)

RESULTS["p3"] = {
    "S": S, "moments": return_moments, "fit": distribution_fit,
    "greeks": book_greeks, "delta_normal": delta_normal,
    "simulation": simulation_results, "gamma_approx": gamma_approximations,
}
save_results()


# =============================================================================
# Problem 4 - Maximum Sharpe ratio with a short put strategy
# =============================================================================
section("Problem 4 - A Sharpe Ratio That Sells Insurance")

p4 = pd.read_csv(BASE / "problem4.csv")
assets = ["EQ", "BOND", "STK", "PUTW"]
monthly_returns = p4[assets].to_numpy()
monthly_rf = 0.03 / 12
mean_returns = monthly_returns.mean(axis=0)
covariance = np.cov(monthly_returns, rowvar=False)
asset_sd = np.sqrt(np.diag(covariance))

# The annual Sharpe ratio uses monthly mean and standard deviation.
asset_stats = {}
for i, asset in enumerate(assets):
    values = rl.sample_moments(p4[asset])
    values["annual_sharpe"] = ((mean_returns[i] - monthly_rf)
                                / asset_sd[i] * np.sqrt(12))
    values["mean_se"] = asset_sd[i] / np.sqrt(len(p4))
    asset_stats[asset] = values
print("Asset moments, Sharpe and mean standard errors:\n", pd.DataFrame(asset_stats).T)

worst_month = p4.loc[p4.PUTW.idxmin(), ["Month", "PUTW", "EQ"]].to_dict()
print("Worst PUTW month:", worst_month)


def annual_sharpe(weights, means=mean_returns, cov=covariance):
    """Annualized Sharpe ratio of a portfolio of monthly returns."""
    expected = weights @ means
    volatility = np.sqrt(weights @ cov @ weights)
    return (expected - monthly_rf) / volatility * np.sqrt(12)


def maximize_sharpe(bounds, start=None):
    """Find long-only tangency weights, optionally limiting PUTW to 20%."""
    if start is None:
        start = np.ones(4) / 4
    fitted = minimize(
        lambda w: -annual_sharpe(w), start, method="SLSQP", bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: sum(w) - 1}],
        options={"ftol": 1e-13, "maxiter": 10000},
    )
    if not fitted.success:
        print("Optimization warning:", fitted.message)
    return fitted.x


# Unconstrained solution uses inverse covariance (no short-selling bounds).
direction = np.linalg.solve(covariance, mean_returns - monthly_rf)
unconstrained = direction / direction.sum()
long_only = maximize_sharpe([(0, 1)] * 4)
put_cap = maximize_sharpe([(0, 1)] * 3 + [(0, 0.20)])
portfolio_weights = {
    "Unconstrained": unconstrained,
    "LongOnly": long_only,
    "PutCap20": put_cap,
}

portfolio_results = {}
for name, weights in portfolio_weights.items():
    monthly_pnl = monthly_returns @ weights  # Returns for a $1 portfolio
    portfolio_results[name] = {
        "weights": dict(zip(assets, weights)),
        "annual_sharpe": annual_sharpe(weights),
        **rl.portfolio_risk_statistics(monthly_pnl),
    }
    print(f"\n{name} weights:", dict(zip(assets, weights)))
    print("Annual Sharpe:", portfolio_results[name]["annual_sharpe"])
    print("Monthly historical/normal risk:",
          {k: v for k, v in portfolio_results[name].items() if "VaR" in k or "ES" in k})

# CAPM: regress PUTW excess returns on EQ excess returns.
put_excess = p4.PUTW - monthly_rf
eq_excess = p4.EQ - monthly_rf
capm_model = sm.OLS(put_excess, sm.add_constant(eq_excess)).fit()
tail_months = eq_excess < -0.05
tail_model = sm.OLS(put_excess[tail_months],
                    sm.add_constant(eq_excess[tail_months])).fit()
capm_results = {
    "alpha_monthly": capm_model.params.iloc[0],
    "alpha_t": capm_model.tvalues.iloc[0],
    "beta": capm_model.params.iloc[1],
    "r2": capm_model.rsquared,
    "tail_beta": tail_model.params.iloc[1],
    "tail_n": int(tail_months.sum()),
    "tail_alpha": tail_model.params.iloc[0],
}
print("\nCAPM and downside beta:", capm_results)

# Construct the long-only efficient frontier: minimum variance at each target mean.
frontier = []
for target_return in np.linspace(mean_returns.min(), mean_returns.max(), 110):
    fitted = minimize(
        lambda w: w @ covariance @ w, np.ones(4) / 4,
        method="SLSQP", bounds=[(0, 1)] * 4,
        constraints=[
            {"type": "eq", "fun": lambda w: sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ mean_returns - target_return},
        ],
        options={"ftol": 1e-13, "maxiter": 5000},
    )
    if fitted.success and abs(fitted.x @ mean_returns - target_return) < 1e-6:
        risk_annual = np.sqrt(fitted.x @ covariance @ fitted.x) * np.sqrt(12)
        frontier.append((risk_annual, target_return * 12))

frontier = np.array(frontier)
plt.figure(figsize=(7, 5))
plt.plot(frontier[:, 0], frontier[:, 1], label="Long-only frontier")
max_risk = max(frontier[:, 0]) * 1.15
risk_grid = np.linspace(0, max_risk, 150)
plt.plot(risk_grid, 0.03 + annual_sharpe(long_only) * risk_grid,
         "--", label="Capital market line")
for i, asset in enumerate(assets):
    plt.scatter(asset_sd[i] * np.sqrt(12), mean_returns[i] * 12, label=asset)
plt.xlabel("Annualized volatility")
plt.ylabel("Annualized arithmetic return")
plt.title("Problem 4: Efficient Frontier and CML")
plt.xlim(0, max_risk)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "p4_frontier.png", dpi=150)
plt.close()

# Check how sensitive long-only weights are to the historical sample.
rng = np.random.default_rng(54504)
bootstrap_weights = []
for repeat in range(250):
    sample = monthly_returns[rng.integers(0, len(monthly_returns), len(monthly_returns))]
    sample_mean = sample.mean(axis=0)
    sample_cov = np.cov(sample, rowvar=False)
    fitted = minimize(
        lambda w: -(w @ sample_mean - monthly_rf)
                  / np.sqrt(max(w @ sample_cov @ w, 1e-20)),
        long_only, method="SLSQP", bounds=[(0, 1)] * 4,
        constraints=[{"type": "eq", "fun": lambda w: sum(w) - 1}],
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if fitted.success:
        bootstrap_weights.append(fitted.x)

bootstrap_weights = np.array(bootstrap_weights)
bootstrap_summary = {}
for i, asset in enumerate(assets):
    bootstrap_summary[asset] = {
        "median": np.median(bootstrap_weights[:, i]),
        "p05": np.quantile(bootstrap_weights[:, i], 0.05),
        "p95": np.quantile(bootstrap_weights[:, i], 0.95),
    }
print("\nBootstrap weight stability:\n", pd.DataFrame(bootstrap_summary).T)

RESULTS["p4"] = {
    "moments": asset_stats, "putw_worst": worst_month,
    "portfolios": portfolio_results, "capm": capm_results,
    "bootstrap": bootstrap_summary,
}
save_results()


# =============================================================================
# Problem 5 - Alpha or factor exposure?
# =============================================================================
section("Problem 5 - Alpha or Factor Exposure")

p5_funds = pd.read_csv(BASE / "problem5.csv")
ff3 = pd.read_csv(BASE / "F-F_Research_Data_Factors_daily.CSV")
momentum = pd.read_csv(BASE / "F-F_Momentum_Factor_daily.CSV")

# Clean names, parse dates, and convert Fama-French percentages to decimals.
ff3.columns = ff3.columns.str.strip()
momentum.columns = momentum.columns.str.strip()
ff3.Date = pd.to_datetime(ff3.Date.astype(str), format="%Y%m%d")
momentum.Date = pd.to_datetime(momentum.Date.astype(str), format="%Y%m%d")
p5_funds.Date = pd.to_datetime(p5_funds.Date)
ff3[["Mkt-RF", "SMB", "HML", "RF"]] /= 100
momentum[["Mom"]] /= 100

factors = ff3.merge(momentum, on="Date", how="inner")
joined = p5_funds.merge(factors, on="Date", how="inner")
factor_names = ["Mkt-RF", "SMB", "HML", "Mom"]
regression_factors = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart4": factor_names,
}
print("Aligned sample days:", len(joined))

# Use the final ten years of common factor data for forward expected returns.
cutoff = factors.Date.max() - pd.DateOffset(years=10)
last_ten_years = factors[factors.Date > cutoff]
expected_factors = last_ten_years[factor_names + ["RF"]].mean()
print("Recent ten-year factor means:\n", expected_factors)

# Worst historical momentum shock, applied with all other factors flat.
worst_momentum = momentum.loc[momentum.Mom.idxmin(), ["Date", "Mom"]]
fund_results = {}

for fund in ["FUND1", "FUND2"]:
    excess_return = joined[fund] - joined.RF
    fund_stats = rl.sample_moments(excess_return)
    fund_stats["sharpe_annual"] = (fund_stats["mean"] / fund_stats["stdev"]
                                    * np.sqrt(255))

    models = {}
    for model_name, selected_factors in regression_factors.items():
        X = sm.add_constant(joined[selected_factors])
        model = sm.OLS(excess_return, X).fit()
        k = len(model.params)
        n = len(joined)
        aicc = -2 * model.llf + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        models[model_name] = {
            "coefficients": model.params.to_dict(),
            "se": model.bse.to_dict(),
            "alpha_t": model.tvalues["const"],
            "r2": model.rsquared,
            "aicc": aicc,
        }

    # CAPM residual correlation with the omitted factors.
    capm_residuals = sm.OLS(
        excess_return, sm.add_constant(joined[["Mkt-RF"]])
    ).fit().resid
    residual_correlations = {}
    for factor in ["SMB", "HML", "Mom"]:
        residual_correlations[factor] = np.corrcoef(
            capm_residuals, joined[factor]
        )[0, 1]

    # Expected return from four-factor betas, once excluding and once adding alpha.
    coefficients = models["Carhart4"]["coefficients"]
    predicted_excess = sum(coefficients[f] * expected_factors[f]
                           for f in factor_names)
    annual_without_alpha = 255 * np.log1p(expected_factors["RF"] + predicted_excess)
    annual_with_alpha = 255 * np.log1p(
        expected_factors["RF"] + predicted_excess + coefficients["const"]
    )

    # Momentum shock P&L on a $10m fund position.
    shock_loss = -coefficients["Mom"] * worst_momentum.Mom * 10_000_000
    shock_in_standard_deviations = shock_loss / (10_000_000 * fund_stats["stdev"])

    fund_results[fund] = {
        "sample_stats": fund_stats, "models": models,
        "capm_residual_corr": residual_correlations,
        "annual_expected_no_alpha": annual_without_alpha,
        "annual_expected_with_alpha": annual_with_alpha,
        "realized_annual_geom": 255 * np.mean(np.log1p(joined[fund])),
        "momentum_shock_loss_dollars": shock_loss,
        "shock_sigma_multiple": shock_in_standard_deviations,
    }
    print(f"\n{fund} excess-return statistics:", fund_stats)
    for model_name, model_result in models.items():
        print(f"{model_name}: coefficients={model_result['coefficients']}, "
              f"SE={model_result['se']}, alpha t={model_result['alpha_t']:.3f}, "
              f"R2={model_result['r2']:.4f}, AICc={model_result['aicc']:.3f}")
    print("CAPM residual correlations:", residual_correlations)
    print("Expected annual return without/with alpha:",
          annual_without_alpha, annual_with_alpha)
    print("Momentum shock loss:", money(shock_loss),
          "; multiple of daily SD:", shock_in_standard_deviations)

RESULTS["p5"] = {
    "n_merged": len(joined),
    "sample_factor_means": joined[factor_names].mean().to_dict(),
    "recent_factor_means": expected_factors.to_dict(),
    "worst_mom": {"Date": str(worst_momentum.Date.date()),
                  "Mom": worst_momentum.Mom},
    "funds": fund_results,
}
save_results()

print("\nDone. Outputs are in:", BASE)
print("Plots are in:", PLOTS)
