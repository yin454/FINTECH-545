"""FINTECH 545 - Functional tests 8.1 through 13.9."""

from pathlib import Path
import numpy as np
import pandas as pd

from risklib import (
    normal_var,
    t_var,
    simulate_t,
    simulated_var,
    normal_es,
    t_es,
    simulated_es,
)

BASE = Path(__file__).resolve().parent
ALPHA = 0.05


def read_csv(name):
    return pd.read_csv(BASE / name)


def assert_values_close(actual, expected, atol=1e-5, rtol=1e-5):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert np.allclose(actual, expected, atol=atol, rtol=rtol)


# =========================================================
# Test 8: VaR and Expected Shortfall
# =========================================================

def test_8_1_var_from_normal():
    data = read_csv("test7_1.csv")
    expected = read_csv("testout8_1.csv").iloc[0]

    actual = normal_var(data["x1"], ALPHA)
    target = expected[["VaR Absolute", "VaR Diff from Mean"]].to_numpy(dtype=float)
    assert_values_close(actual, target)


def test_8_2_var_from_t():
    data = read_csv("test7_2.csv")
    expected = read_csv("testout8_2.csv").iloc[0]

    actual = t_var(data["x1"], ALPHA)
    target = expected[["VaR Absolute", "VaR Diff from Mean"]].to_numpy(dtype=float)
    assert_values_close(actual, target)


def test_8_3_var_from_simulation():
    data = read_csv("test7_2.csv")
    expected = read_csv("testout8_3.csv").iloc[0]

    simulations = simulate_t(data["x1"], n_sim=100000, seed=1234)
    actual = simulated_var(simulations, ALPHA)
    target = expected[["VaR Absolute", "VaR Diff from Mean"]].to_numpy(dtype=float)

    # Simulated results vary with the random seed and simulation sample size.
    assert_values_close(actual, target, atol=0.005, rtol=0)
    assert_values_close(actual, t_var(data["x1"], ALPHA), atol=0.005, rtol=0)


def test_8_4_es_from_normal():
    data = read_csv("test7_1.csv")
    expected = read_csv("testout8_4.csv").iloc[0]

    actual = normal_es(data["x1"], ALPHA)
    target = expected[["ES Absolute", "ES Diff from Mean"]].to_numpy(dtype=float)
    assert_values_close(actual, target)


def test_8_5_es_from_t():
    data = read_csv("test7_2.csv")
    expected = read_csv("testout8_5.csv").iloc[0]

    actual = t_es(data["x1"], ALPHA)
    target = expected[["ES Absolute", "ES Diff from Mean"]].to_numpy(dtype=float)
    assert_values_close(actual, target)


def test_8_6_es_from_simulation():
    data = read_csv("test7_2.csv")
    expected = read_csv("testout8_6.csv").iloc[0]

    simulations = simulate_t(data["x1"], n_sim=100000, seed=1234)
    actual = simulated_es(simulations, ALPHA)
    target = expected[["ES Absolute", "ES Diff from Mean"]].to_numpy(dtype=float)

    # Monte Carlo ES is not identical to an analytic t-distribution ES.
    assert_values_close(actual, target, atol=0.005, rtol=0)
    assert_values_close(actual, t_es(data["x1"], ALPHA), atol=0.005, rtol=0)


# =========================================================
# Test 9: Gaussian Copula Portfolio VaR / ES
# =========================================================

def test_9_1_copula_portfolio_var_es():
    from risklib import gaussian_copula_portfolio_risk

    portfolio = read_csv("test9_1_portfolio.csv")
    returns = read_csv("test9_1_returns.csv")
    expected = read_csv("testout9_1.csv")

    actual = gaussian_copula_portfolio_risk(
        returns, portfolio, n_sim=300000, seed=1234, alpha=0.05
    )

    assert actual["Stock"].tolist() == expected["Stock"].tolist()

    # Monte Carlo results have sampling error, so use dollar and percent tolerances.
    assert_values_close(actual[["VaR95", "ES95"]],
                        expected[["VaR95", "ES95"]], atol=4.0, rtol=0)
    assert_values_close(actual[["VaR95_Pct", "ES95_Pct"]],
                        expected[["VaR95_Pct", "ES95_Pct"]], atol=0.001, rtol=0)


# =========================================================
# Test 10: Risk Parity and Maximum Sharpe Ratio
# =========================================================

def test_10_1_equal_risk_parity():
    from risklib import risk_parity_weights

    covariance = read_csv("test5_2.csv")
    expected = read_csv("testout10_1.csv")["W"]

    actual = risk_parity_weights(covariance)
    assert_values_close(actual, expected, atol=1e-6, rtol=0)


def test_10_2_half_risk_budget_for_x5():
    from risklib import risk_parity_weights

    covariance = read_csv("test5_2.csv")
    expected = read_csv("testout10_2.csv")["W"]

    # X5 receives half the risk budget of each other asset.
    budgets = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
    actual = risk_parity_weights(covariance, budgets)
    assert_values_close(actual, expected, atol=1e-6, rtol=0)


def test_10_3_maximum_sharpe_nonnegative_weights():
    from risklib import max_sharpe_weights

    covariance = read_csv("test5_2.csv")
    means = read_csv("test10_3_means.csv")["Mean"]
    expected = read_csv("testout10_3.csv")["W"]

    actual = max_sharpe_weights(means, covariance, rf=0.04, lower=0.0, upper=1.0)
    # The reference solver allows tiny negative numerical errors (~1e-8).
    assert_values_close(actual, expected, atol=1e-6, rtol=0)


def test_10_4_maximum_sharpe_bounded_weights():
    from risklib import max_sharpe_weights

    covariance = read_csv("test5_2.csv")
    means = read_csv("test10_3_means.csv")["Mean"]
    expected = read_csv("testout10_4.csv")["W"]

    actual = max_sharpe_weights(means, covariance, rf=0.04, lower=0.1, upper=0.5)
    assert_values_close(actual, expected, atol=1e-6, rtol=0)


# =========================================================
# Test 11: Ex-post attribution
# =========================================================

def test_11_1_ex_post_attribution():
    from risklib import ex_post_attribution

    returns = read_csv("test11_1_returns.csv")
    weights = read_csv("test11_1_weights.csv")["W"]
    expected = read_csv("testout11_1.csv")

    actual = ex_post_attribution(returns, weights)
    assert actual["Value"].tolist() == expected["Value"].tolist()
    assert_values_close(actual.drop(columns="Value"),
                        expected.drop(columns="Value"), atol=1e-8, rtol=1e-8)


def test_11_2_ex_post_factor_attribution():
    from risklib import ex_post_factor_attribution

    factors = read_csv("test11_2_factor_returns.csv")
    stocks = read_csv("test11_2_stock_returns.csv")
    beta = read_csv("test11_2_beta.csv")
    weights = read_csv("test11_2_weights.csv")["W"]
    expected = read_csv("testout11_2.csv")

    actual = ex_post_factor_attribution(stocks, factors, beta, weights)
    assert actual["Value"].tolist() == expected["Value"].tolist()
    assert_values_close(actual.drop(columns="Value"),
                        expected.drop(columns="Value"), atol=1e-8, rtol=1e-8)


# =========================================================
# Test 12: European and American option pricing and Greeks
# =========================================================

def test_12_1_european_gbsm_with_greeks():
    from risklib import gbsm_greeks

    data = read_csv("test12_1.csv").dropna(how="all")
    expected = read_csv("testout12_1.csv")

    for _, option in data.iterrows():
        maturity = option["DaysToMaturity"] / option["DayPerYear"]
        actual = gbsm_greeks(
            option["Option Type"], option["Underlying"], option["Strike"],
            maturity, option["RiskFreeRate"], option["DividendRate"], option["ImpliedVol"]
        )
        target = expected.loc[expected["ID"] == option["ID"],
                              ["Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]].iloc[0]
        assert_values_close(actual, target, atol=1e-7, rtol=1e-7)


def test_12_2_american_continuous_dividends_and_greeks():
    from risklib import american_option_greeks

    data = read_csv("test12_1.csv").dropna(how="all")
    expected = read_csv("testout12_2.csv")

    for _, option in data.iterrows():
        maturity = option["DaysToMaturity"] / option["DayPerYear"]
        actual = american_option_greeks(
            option["Option Type"], option["Underlying"], option["Strike"],
            maturity, option["RiskFreeRate"], option["DividendRate"], option["ImpliedVol"],
            steps=500
        )
        target = expected.loc[expected["ID"] == option["ID"],
                              ["Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]].iloc[0]
        # Finite differences on a binomial tree are sensitive to bump size,
        # particularly the rate and time Greeks near an exercise boundary.
        assert_values_close(actual, target, atol=1e-3, rtol=0)


def test_12_3_american_discrete_dividends():
    from risklib import american_option_discrete_dividends

    data = read_csv("test12_3.csv")
    expected = read_csv("testout12_3.csv")

    for _, option in data.iterrows():
        maturity = option["DaysToMaturity"] / option["DayPerYear"]
        dividend_dates = [float(x) for x in option["DividendDates"].split(",")]
        dividend_amounts = [float(x) for x in option["DividendAmts"].split(",")]
        actual = american_option_discrete_dividends(
            option["Option Type"], option["Underlying"], option["Strike"],
            maturity, option["RiskFreeRate"], option["ImpliedVol"],
            dividend_dates, dividend_amounts, steps=500
        )
        target = expected.loc[expected["ID"] == option["ID"], "Value"].iloc[0]
        assert np.isclose(actual, target, atol=1e-5, rtol=0)


# =========================================================
# Test 13: Multivariate t and copulas
# =========================================================

def test_13_1_kendall_correlation():
    from risklib import kendall_correlation
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_1.csv")
    actual = kendall_correlation(data)
    assert_values_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_13_2_multivariate_t_mean():
    from risklib import fit_multivariate_t
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_2.csv")
    mean, scale, nu, ll = fit_multivariate_t(data)
    assert_values_close(mean, expected["mu"], atol=1e-10, rtol=1e-10)


def test_13_3_multivariate_t_scale():
    from risklib import fit_multivariate_t
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_3.csv")
    mean, scale, nu, ll = fit_multivariate_t(data)
    assert_values_close(scale, expected, atol=1e-8, rtol=1e-3)


def test_13_4_multivariate_t_profile_likelihood():
    from risklib import fit_multivariate_t
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_4.csv").iloc[0]
    mean, scale, nu, ll = fit_multivariate_t(data)
    assert np.isclose(nu, expected["nu"], atol=0.001)
    assert np.isclose(ll, expected["ll"], atol=1e-5)


def test_13_5_gaussian_copula_loglik():
    from risklib import kendall_correlation, fitted_t_uniforms, gaussian_copula_loglik
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_5.csv").iloc[0]["ll"]
    correlation = kendall_correlation(data)
    uniforms, fits = fitted_t_uniforms(data)
    actual = gaussian_copula_loglik(uniforms, correlation)
    assert np.isclose(actual, expected, atol=0.001)


def test_13_6_t_copula_profile_likelihood():
    from risklib import kendall_correlation, fitted_t_uniforms, fit_t_copula
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_6.csv").iloc[0]
    correlation = kendall_correlation(data)
    uniforms, fits = fitted_t_uniforms(data)
    nu, ll = fit_t_copula(uniforms, correlation)
    assert np.isclose(nu, expected["nu"], atol=0.001)
    assert np.isclose(ll, expected["ll"], atol=0.001)


def test_13_7_copula_aicc_bic():
    from risklib import (kendall_correlation, fitted_t_uniforms,
                         gaussian_copula_loglik, fit_t_copula, copula_information_criteria)
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_7.csv")
    correlation = kendall_correlation(data)
    uniforms, fits = fitted_t_uniforms(data)
    gaussian_ll = gaussian_copula_loglik(uniforms, correlation)
    t_nu, t_ll = fit_t_copula(uniforms, correlation)
    rows = []
    for name, ll, k in [("Gaussian", gaussian_ll, 0), ("T", t_ll, 1)]:
        aicc, bic = copula_information_criteria(ll, k, len(data))
        rows.append([name, ll, k, aicc, bic])
    actual = pd.DataFrame(rows, columns=["Copula", "LL", "K", "AICC", "BIC"])
    assert actual["Copula"].tolist() == expected["Copula"].tolist()
    assert_values_close(actual.drop(columns="Copula"), expected.drop(columns="Copula"), atol=0.003)


def test_13_8_lower_tail_dependence():
    from risklib import (kendall_correlation, fitted_t_uniforms,
                         fit_t_copula, t_copula_tail_dependence)
    data = read_csv("test13_returns.csv")
    expected = read_csv("testout13_8.csv")
    correlation = kendall_correlation(data)
    uniforms, fits = fitted_t_uniforms(data)
    nu, ll = fit_t_copula(uniforms, correlation)
    actual = t_copula_tail_dependence(correlation, nu)
    assert_values_close(actual, expected, atol=5e-5)


def test_13_9_t_copula_portfolio_risk():
    from risklib import t_copula_portfolio_risk
    data = read_csv("test13_returns.csv")
    portfolio = read_csv("test13_portfolio.csv")
    expected = read_csv("testout13_9.csv")
    actual = t_copula_portfolio_risk(data, portfolio, n_sim=300000, seed=1234)
    assert actual["Stock"].tolist() == expected["Stock"].tolist()
    # Simulation results have random sampling error, unlike analytic results.
    assert_values_close(actual[["VaR95", "ES95"]], expected[["VaR95", "ES95"]], atol=2.5, rtol=0)
    assert_values_close(actual[["VaR95_Pct", "ES95_Pct"]],
                        expected[["VaR95_Pct", "ES95_Pct"]], atol=0.002, rtol=0)
