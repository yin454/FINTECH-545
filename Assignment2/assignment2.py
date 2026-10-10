"""FINTECH 545 - Assignment 2
Covariance, VaR, and Copulas

Run from the Assignment2 directory:
    python assignment2.py

The script prints every reported number and writes the required plots to plots/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import risklib as rl

BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)
ALPHA = 0.05


def money(x):
    return f"${x:,.2f}"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# Problem 1 - Correlations from mismatched histories
# =============================================================================
section("Problem 1 - Correlations from Mismatched Histories")
p1 = pd.read_csv(BASE / "problem1.csv")
r1 = p1[["A", "B", "C", "D", "IDX"]]

counts = rl.pairwise_observation_counts(r1)
all_five = len(r1.dropna())
print("Pairwise joint observation counts:\n", counts)
print("Days on which all five trade:", all_five)

complete_corr = rl.correlation_matrix(r1, skip_miss=True)
pairwise_corr = rl.correlation_matrix(r1, skip_miss=False)
print("\nComplete-case correlation:\n", complete_corr)
print("\nPairwise correlation:\n", pairwise_corr)

for name, corr in [("Complete case", complete_corr), ("Pairwise", pairwise_corr)]:
    eig = np.linalg.eigvalsh(corr)
    print(f"\n{name} eigenvalues:", eig)
    try:
        np.linalg.cholesky(corr.to_numpy())
        print(f"{name} Cholesky: SUCCESS")
    except np.linalg.LinAlgError:
        print(f"{name} Cholesky: FAIL (matrix is not positive definite)")

# Build covariance from pairwise correlations and each series' full-history SD.
sd_full = r1.std(ddof=1)
cov_pairwise = pairwise_corr.to_numpy() * np.outer(sd_full, sd_full)
# Portfolio: long 1 IDX, short 0.40 A, 0.30 B, 0.20 C, 0.10 D.
w_tracking = np.array([-0.40, -0.30, -0.20, -0.10, 1.00])
tracking_var_raw = w_tracking @ cov_pairwise @ w_tracking
print("\nFull-history standard deviations:\n", sd_full)
print("Raw pairwise tracking-portfolio variance:", tracking_var_raw)

rj = rl.near_psd(pairwise_corr)
higham = rl.higham_nearest_psd(pairwise_corr, tol=1e-12, max_iter=10000)

for name, repaired in [("Rebonato-Jaeckel", rj), ("Higham", higham)]:
    eig_min = np.min(np.linalg.eigvalsh(repaired))
    dist = np.linalg.norm(repaired.to_numpy() - pairwise_corr.to_numpy(), ord="fro") #Frobenius distance
    cov_repaired = repaired.to_numpy() * np.outer(sd_full, sd_full)
    track_var = w_tracking @ cov_repaired @ w_tracking
    print(f"{name}: min eigenvalue={eig_min:.12g}, Frobenius distance={dist:.8f}, "
          f"tracking variance={track_var:.12g}")

# Largest absolute entry changes under Higham.
delta_h = higham - pairwise_corr
pairs = []
for i in range(len(r1.columns)):
    for j in range(i + 1, len(r1.columns)):
        pairs.append((abs(delta_h.iloc[i, j]), r1.columns[i], r1.columns[j],
                      delta_h.iloc[i, j], counts.iloc[i, j]))
pairs.sort(reverse=True)
print("\nLargest Higham changes:")
for abs_move, c1, c2, signed_move, nobs in pairs[:5]:
    print(f"{c1}-{c2}: change={signed_move:+.6f}, |change|={abs_move:.6f}, n={nobs}")
print("Frobenius gap complete-case vs pairwise:",
      np.linalg.norm(complete_corr.to_numpy() - pairwise_corr.to_numpy(), ord="fro"))


# =============================================================================
# Problem 2 - Volatility estimate after the regime changed
# =============================================================================
section("Problem 2 - A Volatility Estimate After the Regime Changed")
p2 = pd.read_csv(BASE / "problem2.csv")
r2 = p2["Price"].pct_change(fill_method=None).dropna().to_numpy()
r2_centered = r2 - np.mean(r2)
n2 = len(r2_centered)

plt.figure(figsize=(9, 4))
plt.plot(np.arange(1, n2 + 1), r2_centered, linewidth=0.8)
plt.axvline(n2 - 40, linestyle="--", linewidth=1.0)
plt.title("Problem 2: Demeaned Daily Returns")
plt.xlabel("Day")
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(PLOTS / "problem2_returns.png", dpi=180)
plt.close()

m2 = rl.sample_moments(r2_centered)
print("Sample mean removed from returns:", np.mean(r2))
print("First four moments of demeaned returns:", m2)

for lam in [0.94, 0.97]:
    ew_sd = rl.ew_standard_deviation(r2_centered, lam, demean=False)
    neff = rl.effective_sample_size(n2, lam)
    half_life = rl.ew_half_life(lam)
    recent_weight = rl.exp_weights(n2, lam)[-40:].sum()
    se_vol = ew_sd / np.sqrt(2.0 * neff) # EWMA 波动率估计值的标准误
    print(f"lambda={lam:.2f}: EW SD={ew_sd:.8f}, n_eff={neff:.4f}, "
          f"half-life={half_life:.4f}, weight on last 40 days={recent_weight:.4%}, "
          f"vol SE~{se_vol:.8f}")

position = 1_000_000.0
sd_equal = np.std(r2_centered, ddof=1)
var_equal = -stats.norm.ppf(ALPHA) * sd_equal * position
sd_097 = rl.ew_standard_deviation(r2_centered, 0.97, demean=False)
sd_094 = rl.ew_standard_deviation(r2_centered, 0.94, demean=False)
var_097 = -stats.norm.ppf(ALPHA) * sd_097 * position
var_094 = -stats.norm.ppf(ALPHA) * sd_094 * position

tfit2 = rl.fit_t_mle(r2_centered)
var_t = -stats.t.ppf(ALPHA, df=tfit2["nu"], loc=tfit2["mu"], scale=tfit2["sigma"]) * position
var_hist = -np.quantile(r2_centered, ALPHA) * position

print("\nFive 5% one-day VaR estimates:")
print("Normal equal-weight SD:", money(var_equal))
print("Normal EW lambda=0.97:", money(var_097))
print("Normal EW lambda=0.94:", money(var_094))
print("Student-t MLE:", money(var_t))
print("Historical full sample:", money(var_hist))
print("Student-t fit: mu={:.8f}, scale={:.8f}, nu={:.6f}".format(
    tfit2["mu"], tfit2["sigma"], tfit2["nu"]))

old_sd = np.std(r2_centered[:-40], ddof=1)
recent_sd = np.std(r2_centered[-40:], ddof=1)
print("Old-regime SD (first 460 days):", old_sd)
print("Recent-regime SD (last 40 days):", recent_sd)
print("EW VaR gap (0.94 - 0.97):", money(var_094 - var_097))


# =============================================================================
# Problem 3 - When diversification raises VaR
# =============================================================================
section("Problem 3 - When Diversification Raises VaR")
p3 = pd.read_csv(BASE / "problem3.csv")
A = p3["A"].to_numpy()
B = p3["B"].to_numpy()

# Required P&L distribution plot for concentrated vs diversified choices.
pnl_concentrated = 2_000_000 * A
pnl_diversified = 1_000_000 * (A + B)
plt.figure(figsize=(9, 4))
plt.hist(pnl_concentrated, bins=80, alpha=0.55, density=True, label="$2m in A")
plt.hist(pnl_diversified, bins=80, alpha=0.55, density=True, label="$1m A + $1m B")
plt.title("Problem 3: P&L Distributions")
plt.xlabel("One-Year P&L ($)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "problem3_pnl_distributions.png", dpi=180)
plt.close()

print("A moments:", rl.sample_moments(A))
print("B moments:", rl.sample_moments(B))
loss_A = A < -0.20
loss_B = B < -0.20
print("Scenarios loss >20%: A={}, B={}, at least one={}, both={}".format(
    loss_A.sum(), loss_B.sum(), np.logical_or(loss_A, loss_B).sum(),
    np.logical_and(loss_A, loss_B).sum()))

p3_positions = {
    "$1m A": 1_000_000 * A,
    "$1m B": 1_000_000 * B,
    "$2m A": 2_000_000 * A,
    "$1m A + $1m B": 1_000_000 * (A + B),
}

for alpha in [0.05, 0.01]:
    print(f"\nHistorical VaR/ES at alpha={alpha:.0%}:")
    for name, pnl in p3_positions.items():
        v, es = rl.historical_var_es(pnl, alpha)
        print(f"{name}: VaR={money(v)}, ES={money(es)}")

print("\nNormal 5% VaR:")
for name, pnl in p3_positions.items():
    v = rl.normal_var(np.mean(pnl), np.std(pnl, ddof=1), 0.05)
    print(f"{name}: {money(v)}")

vA, esA = rl.historical_var_es(p3_positions["$1m A"], 0.05)
vB, esB = rl.historical_var_es(p3_positions["$1m B"], 0.05)
vAB, esAB = rl.historical_var_es(p3_positions["$1m A + $1m B"], 0.05)
print("\nSubadditivity check at 5%:")
print(f"VaR(A)+VaR(B)={money(vA + vB)} vs VaR(A+B)={money(vAB)}")
print(f"ES(A)+ES(B)={money(esA + esB)} vs ES(A+B)={money(esAB)}")


# =============================================================================
# Problem 4 - Gaussian or t Copula
# =============================================================================
section("Problem 4 - Gaussian or t Copula")
p4 = pd.read_csv(BASE / "problem4.csv")
cols4 = list(p4.columns)

for c in cols4:
    moments = rl.sample_moments(p4[c])
    out_idx = np.argmax(np.abs(p4[c] - p4[c].mean()))
    without = p4[c].drop(index=out_idx)
    print(f"{c} moments: {moments}")
    print(f"  most extreme observation={p4[c].iloc[out_idx]:.8f}; "
          f"excess kurtosis without it={rl.sample_moments(without)['excess_kurtosis']:.6f}")

rank_u = rl.rank_uniforms(p4)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
pair_list = [(0, 1), (0, 2), (1, 2)]
for ax, (i, j) in zip(axes, pair_list):
    ax.scatter(rank_u.iloc[:, i], rank_u.iloc[:, j], s=8, alpha=0.45)
    ax.set_xlabel(cols4[i])
    ax.set_ylabel(cols4[j])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.suptitle("Problem 4: Rank-Scaled Pair Plots")
fig.tight_layout()
fig.savefig(PLOTS / "problem4_rank_pairs.png", dpi=180)
plt.close(fig)

print("\nObserved joint 2.5% tail-day counts (independence expectation = 0.625):")
for i, j in pair_list:
    lo = ((rank_u.iloc[:, i] <= 0.025) & (rank_u.iloc[:, j] <= 0.025)).sum()
    hi = ((rank_u.iloc[:, i] >= 0.975) & (rank_u.iloc[:, j] >= 0.975)).sum()
    print(f"{cols4[i]}-{cols4[j]}: lower={lo}, upper={hi}")

# Fit Normal and t margins and choose by AICc.
margin_fits = {}
print("\nMarginal fits:")
for c in cols4:
    chosen, norm_fit, t_fit = rl.choose_normal_or_t(p4[c])
    margin_fits[c] = chosen
    print(f"{c}: Normal AICc={norm_fit['aicc']:.6f}; "
          f"t AICc={t_fit['aicc']:.6f}, t nu={t_fit['nu']:.6f}; "
          f"chosen={chosen['distribution']}")

# Transform observations to uniforms through selected fitted margins.
Ufit = np.column_stack([
    rl.fitted_cdf(p4[c].to_numpy(), margin_fits[c]) for c in cols4
])
Ufit = np.clip(Ufit, 1e-12, 1 - 1e-12)
R4 = rl.kendall_tau_correlation(p4)
print("\nKendall-tau-implied correlation matrix:\n", R4)

ll_gauss = rl.gaussian_copula_loglik(Ufit, R4)
t_cop = rl.fit_t_copula(Ufit, R4)
nu_cop = t_cop["nu"]
ll_tcop = t_cop["log_likelihood"]
# Week 5 Section 6.5: margins and the common Kendall-tau R are frozen.
# Therefore Gaussian has k=0 free copula parameters and t has only k=1 (nu).
print("Gaussian copula: LL={:.6f}, AICc={:.6f}, BIC={:.6f}".format(
    ll_gauss, rl.aicc(ll_gauss, 0, len(p4)), rl.bic(ll_gauss, 0, len(p4))))
print("t copula: nu={:.6f}, LL={:.6f}, AICc={:.6f}, BIC={:.6f}".format(
    nu_cop, ll_tcop, rl.aicc(ll_tcop, 1, len(p4)), rl.bic(ll_tcop, 1, len(p4))))

# Simulate from both copulas through the selected margins.
def uniforms_to_returns(U):
    out = np.empty_like(U)
    for j, c in enumerate(cols4):
        out[:, j] = rl.fitted_ppf(U[:, j], margin_fits[c])
    return out

sim_results = {}
for copula, U in [
    ("Gaussian", rl.simulate_gaussian_copula(R4, 100_000, seed=20261007)),
    ("t", rl.simulate_t_copula(R4, nu_cop, 100_000, seed=20261008)),
]:
    sim_ret = uniforms_to_returns(U)
    pnl = 1_000_000 * sim_ret.sum(axis=1)
    sim_results[copula] = (U, pnl)
    print(f"\n{copula} copula portfolio risk:")
    for alpha in [0.05, 0.01]:
        v, es = rl.historical_var_es(pnl, alpha)
        print(f"alpha={alpha:.0%}: VaR={money(v)}, ES={money(es)}")
    print("Expected joint tail days per 1,000:")
    for i, j in pair_list:
        lo = 1000 * np.mean((U[:, i] <= 0.025) & (U[:, j] <= 0.025))
        hi = 1000 * np.mean((U[:, i] >= 0.975) & (U[:, j] >= 0.975))
        print(f"{cols4[i]}-{cols4[j]}: lower={lo:.2f}, upper={hi:.2f}")

hist_pnl4 = 1_000_000 * p4.sum(axis=1).to_numpy()
print("\nHistorical portfolio risk:")
for alpha in [0.05, 0.01]:
    v, es = rl.historical_var_es(hist_pnl4, alpha)
    print(f"alpha={alpha:.0%}: VaR={money(v)}, ES={money(es)}")

# Day-by-day copula log-likelihood difference.
llg_i = rl.gaussian_copula_loglik(Ufit, R4, return_contributions=True)
llt_i = rl.t_copula_loglik(Ufit, R4, nu_cop, return_contributions=True)
ll_diff = llt_i - llg_i
middle = ((Ufit > 0.05) & (Ufit < 0.95)).all(axis=1)
print("\nLL(t)-LL(Gaussian) total:", ll_diff.sum())
print("Contribution from days with no series in outer 5%:", ll_diff[middle].sum())
print("Fraction from those middle days:", ll_diff[middle].sum() / ll_diff.sum())

# Tail dependence for the most correlated pair.
R4_np = R4.to_numpy()
upper_triangle = np.triu(R4_np, k=1)
i_max, j_max = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
rho_max = R4_np[i_max, j_max]
lambda_t = rl.t_copula_tail_dependence(rho_max, nu_cop)
print(f"Most correlated pair: {cols4[i_max]}-{cols4[j_max]}, rho={rho_max:.6f}")
print(f"t-copula tail dependence={lambda_t:.6f}; Gaussian tail dependence=0")


# =============================================================================
# Problem 5 - Model based simulation and residual correlation
# =============================================================================
section("Problem 5 - Model Based Simulation and Residual Correlation")
p5 = pd.read_csv(BASE / "problem5.csv")
ret5 = p5[["MKT", "A", "B"]].pct_change(fill_method=None).dropna().reset_index(drop=True)

fit_A = rl.ols_factor_regression(ret5["MKT"], ret5["A"])
fit_B = rl.ols_factor_regression(ret5["MKT"], ret5["B"])
resid_mat = np.column_stack([fit_A["residuals"], fit_B["residuals"]])
resid_cov = np.cov(resid_mat, rowvar=False, ddof=1)
resid_corr = np.corrcoef(resid_mat, rowvar=False)[0, 1]

for name, fit in [("A", fit_A), ("B", fit_B)]:
    print(f"{name}: alpha={fit['alpha']:.8f}, beta={fit['beta']:.6f}, "
          f"residual SD={fit['residual_std']:.8f}")
print("Residual correlation:", resid_corr)
print("Residual covariance matrix:\n", resid_cov)

# The prompt says to assume zero expected returns. Therefore the simulations use
# zero means for the market and residuals, and do not add the fitted intercepts.
rng = np.random.default_rng(20261007)
N5 = 100_000
market_sd = np.std(ret5["MKT"], ddof=1)
market_sim = rng.normal(0.0, market_sd, size=N5)

for assumption, cov_e in [
    ("Full residual covariance", resid_cov),
    ("Independent residuals", np.diag(np.diag(resid_cov))),
]:
    e_sim = rng.multivariate_normal(np.zeros(2), cov_e, size=N5)
    A_sim = fit_A["beta"] * market_sim + e_sim[:, 0]
    B_sim = fit_B["beta"] * market_sim + e_sim[:, 1]
    pnl_p1 = 1_000_000 * (A_sim + B_sim)
    pnl_p2 = 1_000_000 * (A_sim - B_sim)
    print(f"\n{assumption}:")
    print("P1 VaR:", money(rl.historical_var_es(pnl_p1, 0.05)[0]))
    print("P2 VaR:", money(rl.historical_var_es(pnl_p2, 0.05)[0]))

# Delta-normal VaR from the sample covariance of stock returns.
stock_cov = ret5[["A", "B"]].cov().to_numpy()
z = -stats.norm.ppf(0.05)
for name, exposure in [
    ("P1", np.array([1_000_000.0, 1_000_000.0])),
    ("P2", np.array([1_000_000.0, -1_000_000.0])),
]:
    sigma_p = np.sqrt(exposure @ stock_cov @ exposure)
    print(f"Delta-normal {name} VaR: {money(z * sigma_p)}")

print("\nDone. Plots were written to:", PLOTS)
