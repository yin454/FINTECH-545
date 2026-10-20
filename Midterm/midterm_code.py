from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import risklib_midterm as rl

BASE = Path(__file__).resolve().parent
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def part(label):
    print("\n" + "-" * 70)
    print(label)
    print("-" * 70)


def money(x):
    return f"${x:,.2f}"

# =========================================================
# Problem 1: Pairwise Correlation with Missing Data
# =========================================================

df = pd.read_csv("problem1.csv")
print("Data shape:", df.shape)
print("Non-missing observations:\n", df.count())


# ---------------------------------------------------------
# Problem 1(a): Pairwise correlation matrix
# ---------------------------------------------------------

pairwise_corr = rl.correlation_matrix(df, skip_miss=False)

print("\nProblem 1(a): Pairwise Correlation Matrix")
print(pairwise_corr.round(6))


# ---------------------------------------------------------
# Problem 1(b): Variance using all available observations
# ---------------------------------------------------------

variances = df.var(ddof=1)
std_devs = np.sqrt(variances)

print("\nProblem 1(b): Sample Variances")
print(variances.round(8))


# ---------------------------------------------------------
# Problem 1(c): Check positive semi-definiteness
# ---------------------------------------------------------

eigenvalues = rl.matrix_eigenvalues(pairwise_corr)
is_psd = np.all(eigenvalues >= -1e-10)

print("\nProblem 1(c): Eigenvalues of Original Correlation Matrix")
print(eigenvalues)
print("Is PSD:", is_psd)


# ---------------------------------------------------------
# Problem 1(d): Higham nearest PSD correlation matrix
# ---------------------------------------------------------

if not is_psd:
    repaired_corr = rl.higham_nearest_psd(pairwise_corr)
else:
    repaired_corr = pairwise_corr.copy()

repaired_eigenvalues = rl.matrix_eigenvalues(repaired_corr)

print("\nProblem 1(d): Higham Repaired Correlation Matrix")
print(repaired_corr.round(6))

print("\nEigenvalues of Repaired Matrix")
print(repaired_eigenvalues)

print("\nIs Repaired Matrix PSD:",
      np.all(repaired_eigenvalues >= -1e-8))

# Changes in correlation coefficients
delta = repaired_corr - pairwise_corr

changes = []
cols = pairwise_corr.columns

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        changes.append({
            "Pair": f"{cols[i]}-{cols[j]}",
            "Original": pairwise_corr.iloc[i, j],
            "Repaired": repaired_corr.iloc[i, j],
            "Change": delta.iloc[i, j],
            "Absolute Change": abs(delta.iloc[i, j])
        })

changes_df = pd.DataFrame(changes)
changes_df = changes_df.sort_values(
    by="Absolute Change", ascending=False
)

print("\nLargest Changes in Correlations")
print(changes_df.to_string(index=False, float_format="%.6f"))


# ---------------------------------------------------------
# Problem 1(e): Construct covariance matrix
# ---------------------------------------------------------

covariance_matrix = repaired_corr.mul(std_devs, axis=0).mul(std_devs, axis=1)

print("\nProblem 1(e): Final Covariance Matrix")
print(covariance_matrix.round(8))

print("\nEigenvalues of Final Covariance Matrix")
print(rl.matrix_eigenvalues(covariance_matrix))

print("\nIs Final Covariance Matrix PSD:",
      np.all(rl.matrix_eigenvalues(covariance_matrix) >= -1e-10))



# =========================================================
# Problem 2: VaR and ES for a Single Stock
# =========================================================

df = pd.read_csv("problem2.csv")

returns = rl.calculate_returns(df, method="arithmetic")

r_spy = returns["SPY"].to_numpy()
r_xyz = returns["XYZ"].to_numpy()

shares = 100
latest_price = df["XYZ"].iloc[-1]
position_value = shares * latest_price

print("Latest XYZ Price:", latest_price)
print("Position Value:", position_value)


# ---------------------------------------------------------
# Problem 2: Fit the Normal factor regression
# r_XYZ = alpha + beta * r_SPY + epsilon
# ---------------------------------------------------------

fit = rl.fit_normal_regression(r_spy, r_xyz)

alpha = fit["alpha"]
beta = fit["beta"]

residuals = r_xyz - (alpha + beta * r_spy)

mu_spy, sigma_spy = rl.fit_normal_mle(r_spy)["mu"], rl.fit_normal_mle(r_spy)["sigma"]
sigma_epsilon = rl.fit_normal_mle(residuals)["sigma"]

mu_xyz = alpha + beta * mu_spy

sigma_xyz = np.sqrt(
    beta**2 * sigma_spy**2 + sigma_epsilon**2
)

print("\nNormal Factor Model Parameters")
print(f"Alpha: {alpha:.8f}")
print(f"Beta: {beta:.8f}")
print(f"SPY Mean: {mu_spy:.8f}")
print(f"SPY Standard Deviation: {sigma_spy:.8f}")
print(f"Residual Standard Deviation: {sigma_epsilon:.8f}")
print(f"XYZ Implied Mean: {mu_xyz:.8f}")
print(f"XYZ Implied Standard Deviation: {sigma_xyz:.8f}")


# ---------------------------------------------------------
# Problem 2(a): VaR at 95% and 99% confidence levels
# ---------------------------------------------------------

confidence_levels = [0.95, 0.99]

results = []

for confidence in confidence_levels:
    tail_alpha = 1 - confidence

    var_pct = rl.normal_var(
        mu_xyz, sigma_xyz, alpha=tail_alpha
    ) * 100

    var_dollar = position_value * var_pct / 100

    results.append({
        "Confidence Level": f"{confidence:.0%}",
        "VaR (%)": var_pct,
        "VaR ($)": var_dollar
    })

var_table = pd.DataFrame(results)

print("\nProblem 2(a): Normal VaR")
print(var_table.to_string(index=False, float_format="%.6f"))


# ---------------------------------------------------------
# Problem 2(b): ES at 95% and 99% confidence levels
# ---------------------------------------------------------

normal_fit = {
    "distribution": "normal",
    "mu": mu_xyz,
    "sigma": sigma_xyz
}

results_es = []

for confidence in confidence_levels:
    tail_alpha = 1 - confidence

    var, es = rl.parametric_var_es(
        normal_fit, alpha=tail_alpha
    )

    results_es.append({
        "Confidence Level": f"{confidence:.0%}",
        "VaR (%)": var * 100,
        "ES (%)": es * 100,
        "VaR ($)": var * position_value,
        "ES ($)": es * position_value
    })

risk_table = pd.DataFrame(results_es)

print("\nProblem 2(b): Normal Expected Shortfall")
print(risk_table.to_string(index=False, float_format="%.6f"))

print("\nMarkdown Table:")
print(risk_table.round(4).to_markdown(index=False))



# =========================================================
# Problem 3: Copula VaR and ES for a Two-Book Portfolio
# =========================================================

SEED = 5452026
N_SIM = 10000
ALPHA = 0.05

prices = pd.read_csv("problem3.csv")
holdings = pd.read_csv("problem3_holdings.csv")

stocks = [f"S{i}" for i in range(1, 11)]

returns = rl.calculate_returns(
    prices, method="arithmetic"
)[stocks].dropna()

latest_prices = prices[stocks].iloc[-1]

holdings = holdings.set_index("Stock").loc[stocks]

position_values = latest_prices * holdings["Holding"]

print("Returns shape:", returns.shape)
print("\nPosition Values:")
print(position_values.round(2))


# =========================================================
# Problem 3(a): Fit Normal, Student-t, and NIG
# =========================================================

def fit_nig_with_aicc(x):
    x = np.asarray(x, dtype=float)

    mu, alpha, beta, delta = rl.fit_nig_mle(x)

    a = alpha * delta
    b = beta * delta

    ll = np.sum(
        stats.norminvgauss.logpdf(
            x, a=a, b=b, loc=mu, scale=delta
        )
    )

    return {
        "distribution": "NIG",
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "log_likelihood": ll,
        "aicc": rl.aicc(ll, k=4, n=len(x))
    }


def distribution_cdf(x, fit):
    if fit["distribution"] in ("normal", "t"):
        return rl.fitted_cdf(x, fit)

    return stats.norminvgauss.cdf(
        x,
        a=fit["alpha"] * fit["delta"],
        b=fit["beta"] * fit["delta"],
        loc=fit["mu"],
        scale=fit["delta"]
    )


def distribution_ppf(u, fit):
    u = np.clip(np.asarray(u), 1e-10, 1 - 1e-10)

    if fit["distribution"] in ("normal", "t"):
        return rl.fitted_ppf(u, fit)

    return stats.norminvgauss.ppf(
        u,
        a=fit["alpha"] * fit["delta"],
        b=fit["beta"] * fit["delta"],
        loc=fit["mu"],
        scale=fit["delta"]
    )


all_fits = {}
selected_fits = {}
fit_rows = []

for stock in stocks:
    x = returns[stock].to_numpy()

    normal_fit = rl.fit_normal_mle(x)
    t_fit = rl.fit_t_mle(x)
    nig_fit = fit_nig_with_aicc(x)

    fits = {
        "normal": normal_fit,
        "t": t_fit,
        "NIG": nig_fit
    }

    all_fits[stock] = fits

    best = min(fits.values(), key=lambda f: f["aicc"])
    selected_fits[stock] = best

    for name, fit in fits.items():
        fit_rows.append({
            "Stock": stock,
            "Distribution": name,
            "mu": fit["mu"],
            "sigma": fit.get("sigma", np.nan),
            "nu": fit.get("nu", np.nan),
            "alpha": fit.get("alpha", np.nan),
            "beta": fit.get("beta", np.nan),
            "delta": fit.get("delta", np.nan),
            "Log Likelihood": fit["log_likelihood"],
            "AICc": fit["aicc"]
        })

fit_table = pd.DataFrame(fit_rows)

print("\nProblem 3(a): Fitted Distribution Parameters")
print(fit_table.to_string(index=False, float_format="%.6f"))

print("\nMarkdown Table:")
print(fit_table.round(4).to_markdown(index=False))

selection_table = pd.DataFrame([
    {
        "Stock": stock,
        "Normal AICc": all_fits[stock]["normal"]["aicc"],
        "Student-t AICc": all_fits[stock]["t"]["aicc"],
        "NIG AICc": all_fits[stock]["NIG"]["aicc"],
        "Selected": selected_fits[stock]["distribution"]
    }
    for stock in stocks
])

print("\nProblem 3(a): Distribution Selection")
print(selection_table.to_string(index=False, float_format="%.4f"))

print("\nMarkdown Table:")
print(selection_table.round(4).to_markdown(index=False))


# =========================================================
# Problem 3(b): Gaussian Copula vs Student-t Copula
# =========================================================

# Transform historical returns using selected marginal CDFs
U = np.column_stack([
    distribution_cdf(
        returns[stock].to_numpy(),
        selected_fits[stock]
    )
    for stock in stocks
])

U = np.clip(U, 1e-10, 1 - 1e-10)

# Estimate copula correlation matrix using Kendall's tau
corr = rl.kendall_tau_correlation(returns)

# Ensure correlation matrix is PSD
if np.min(rl.matrix_eigenvalues(corr)) < -1e-10:
    corr = rl.higham_nearest_psd(corr)

print("\nProblem 3(b): Copula Correlation Matrix")
print(corr.round(4))

print("\nCorrelation Matrix Eigenvalues")
print(rl.matrix_eigenvalues(corr))

# Gaussian copula log likelihood
gaussian_ll = rl.gaussian_copula_loglik(U, corr)

# Student-t copula: fit degrees of freedom
t_copula_fit = rl.fit_t_copula(U, corr)

t_nu = t_copula_fit["nu"]
t_ll = t_copula_fit["log_likelihood"]

# AICc comparison
# 10 stocks -> 45 unique correlation parameters
n = len(U)
d = len(stocks)

k_gaussian = 0
k_t = 1

gaussian_aicc = rl.aicc(gaussian_ll, k_gaussian, n)
t_aicc = rl.aicc(t_ll, k_t, n)

copula_table = pd.DataFrame([
    {
        "Copula": "Gaussian",
        "nu": np.nan,
        "Log Likelihood": gaussian_ll,
        "AICc": gaussian_aicc
    },
    {
        "Copula": "Student-t",
        "nu": t_nu,
        "Log Likelihood": t_ll,
        "AICc": t_aicc
    }
])

selected_copula = (
    "Gaussian"
    if gaussian_aicc <= t_aicc
    else "Student-t"
)

print("\nProblem 3(b): Copula Comparison")
print(copula_table.to_string(index=False, float_format="%.6f"))

print("\nSelected Copula:", selected_copula)

print("\nMarkdown Table:")
print(copula_table.round(4).to_markdown(index=False))


# =========================================================
# Problem 3(c): Simulate Portfolio VaR and ES
# =========================================================

# Generate correlated uniform simulations
if selected_copula == "Gaussian":
    U_sim = rl.simulate_gaussian_copula(
        corr,
        n_sim=N_SIM,
        seed=SEED
    )
else:
    U_sim = rl.simulate_t_copula(
        corr,
        nu=t_nu,
        n_sim=N_SIM,
        seed=SEED
    )

U_sim = np.clip(U_sim, 1e-10, 1 - 1e-10)

# Convert uniform simulations into stock returns
# For NIG, interpolate its inverse CDF for faster simulation

def simulated_quantiles(u, fit):
    if fit["distribution"] != "NIG":
        return distribution_ppf(u, fit)

    grid_u = np.linspace(1e-6, 1 - 1e-6, 1001)

    grid_x = distribution_ppf(grid_u, fit)

    return np.interp(
        u,
        grid_u,
        grid_x
    )


sim_returns = pd.DataFrame({
    stock: simulated_quantiles(
        U_sim[:, i],
        selected_fits[stock]
    )
    for i, stock in enumerate(stocks)
})

print("\nProblem 3(c): Simulated Returns Shape")
print(sim_returns.shape)


# ---------------------------------------------------------
# Calculate stock-level dollar P&L
# ---------------------------------------------------------

stock_pnl = sim_returns.mul(
    position_values,
    axis=1
)

# Book assignments
book_a_stocks = holdings.index[
    holdings["Book"] == "A"
].tolist()

book_b_stocks = holdings.index[
    holdings["Book"] == "B"
].tolist()

# Aggregate P&L
portfolio_pnl = stock_pnl.copy()

portfolio_pnl["Book A"] = stock_pnl[
    book_a_stocks
].sum(axis=1)

portfolio_pnl["Book B"] = stock_pnl[
    book_b_stocks
].sum(axis=1)

portfolio_pnl["Total Portfolio"] = (
    portfolio_pnl["Book A"]
    + portfolio_pnl["Book B"]
)


# ---------------------------------------------------------
# Calculate 95% VaR and ES
# ---------------------------------------------------------

risk_table = rl.risk_table_from_pnl(
    portfolio_pnl,
    alpha=ALPHA
)

risk_table = risk_table.loc[
    stocks + ["Book A", "Book B", "Total Portfolio"]
]

risk_table.index.name = "Position"

print("\nProblem 3(c): 95% VaR and ES ($)")
print(risk_table.to_string(float_format="%.2f"))

print("\nMarkdown Table:")
print(risk_table.round(2).to_markdown())


# ---------------------------------------------------------
# Portfolio Value Summary
# ---------------------------------------------------------

book_a_value = position_values.loc[book_a_stocks].sum()
book_b_value = position_values.loc[book_b_stocks].sum()
total_value = position_values.sum()

value_table = pd.DataFrame({
    "Position": ["Book A", "Book B", "Total Portfolio"],
    "Market Value ($)": [
        book_a_value,
        book_b_value,
        total_value
    ]
})

print("\nPortfolio Market Values")
print(value_table.to_string(index=False, float_format="%.2f"))

print("\nMarkdown Table:")
print(value_table.round(2).to_markdown(index=False))


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

fit_table.to_csv("problem3_distribution_fits.csv", index=False)
selection_table.to_csv("problem3_distribution_selection.csv", index=False)
copula_table.to_csv("problem3_copula_comparison.csv", index=False)
risk_table.to_csv("problem3_var_es.csv")

print("\nProblem 3 completed.")
