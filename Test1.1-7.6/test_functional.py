
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklib import (
    covariance_matrix,
    correlation_matrix,
    ew_covariance,
    ew_correlation,
    ew_covariance_mixed,
    near_psd,
    higham_nearest_psd,
    chol_psd,
    simulate_normal,
    simulate_pca,
    calculate_returns,
    fit_normal,
    fit_t,
    fit_t_regression,
    calculate_aicc_t,
    fit_nig_moments,
    fit_nig_mle,
)


BASE = Path(__file__).resolve().parent


def read_csv(name):
    return pd.read_csv(BASE / name)


def assert_matrix_close(actual, expected, atol=1e-8, rtol=1e-8):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


def covariance_frobenius_error(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return np.linalg.norm(actual - expected, ord="fro")


# =========================================================
# Test 1: Missing data
# =========================================================

def test_1_1_covariance_skip_missing_rows():
    data = read_csv("test1.csv")
    expected = read_csv("testout_1.1.csv")

    actual = covariance_matrix(data, skip_miss=True)
    assert_matrix_close(actual, expected)


def test_1_2_correlation_skip_missing_rows():
    data = read_csv("test1.csv")
    expected = read_csv("testout_1.2.csv")

    actual = correlation_matrix(data, skip_miss=True)
    assert_matrix_close(actual, expected)


def test_1_3_covariance_pairwise():
    data = read_csv("test1.csv")
    expected = read_csv("testout_1.3.csv")

    actual = covariance_matrix(data, skip_miss=False)
    assert_matrix_close(actual, expected)


def test_1_4_correlation_pairwise():
    data = read_csv("test1.csv")
    expected = read_csv("testout_1.4.csv")

    actual = correlation_matrix(data, skip_miss=False)
    assert_matrix_close(actual, expected)


# =========================================================
# Test 2: EW covariance / correlation
# =========================================================

def test_2_1_ew_covariance():
    data = read_csv("test2.csv")
    expected = read_csv("testout_2.1.csv")

    actual = ew_covariance(data, 0.97)
    assert_matrix_close(actual, expected)


def test_2_2_ew_correlation():
    data = read_csv("test2.csv")
    expected = read_csv("testout_2.2.csv")

    actual = ew_correlation(data, 0.94)
    assert_matrix_close(actual, expected)


def test_2_3_mixed_ew_covariance():
    data = read_csv("test2.csv")
    expected = read_csv("testout_2.3.csv")

    actual = ew_covariance_mixed(
        data,
        variance_lambda=0.97,
        correlation_lambda=0.94,
    )
    assert_matrix_close(actual, expected)


# =========================================================
# Test 3: PSD repair
# =========================================================

def test_3_1_near_psd_covariance():
    data = read_csv("testout_1.3.csv")
    expected = read_csv("testout_3.1.csv")

    actual = near_psd(data)
    assert_matrix_close(actual, expected)


def test_3_2_near_psd_correlation():
    data = read_csv("testout_1.4.csv")
    expected = read_csv("testout_3.2.csv")

    actual = near_psd(data)
    assert_matrix_close(actual, expected)


def test_3_3_higham_covariance():
    data = read_csv("testout_1.3.csv")
    expected = read_csv("testout_3.3.csv")

    actual = higham_nearest_psd(data)
    assert_matrix_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_3_4_higham_correlation():
    data = read_csv("testout_1.4.csv")
    expected = read_csv("testout_3.4.csv")

    actual = higham_nearest_psd(data)
    assert_matrix_close(actual, expected, atol=1e-7, rtol=1e-7)


# =========================================================
# Test 4: PSD Cholesky
# =========================================================

def test_4_1_chol_psd():
    data = read_csv("testout_3.1.csv")
    expected = read_csv("testout_4.1.csv")

    actual = chol_psd(data)
    assert_matrix_close(actual, expected)


# =========================================================
# Test 5: Simulation
#
# The provided expected files are sample covariances from random simulations,
# not deterministic theoretical covariances. Therefore these tests compare
# covariance matrices using a practical Frobenius-distance tolerance rather
# than requiring element-for-element equality.
# =========================================================

SIM_TOL = 0.01
SIM_SEED = 1234


def test_5_1_normal_simulation_pd():
    cov = read_csv("test5_1.csv")
    expected = read_csv("testout_5.1.csv")

    sim = simulate_normal(cov, n_sim=100000, mean=np.zeros(cov.shape[0]), seed=SIM_SEED)
    actual = sim.cov()

    assert covariance_frobenius_error(actual, expected) < SIM_TOL


def test_5_2_normal_simulation_psd():
    cov = read_csv("test5_2.csv")
    expected = read_csv("testout_5.2.csv")

    sim = simulate_normal(cov, n_sim=100000, mean=np.zeros(cov.shape[0]), seed=SIM_SEED)
    actual = sim.cov()

    assert covariance_frobenius_error(actual, expected) < SIM_TOL


def test_5_3_normal_simulation_near_psd_fix():
    cov = read_csv("test5_3.csv")
    expected = read_csv("testout_5.3.csv")

    fixed = near_psd(cov)
    sim = simulate_normal(fixed, n_sim=100000, mean=np.zeros(cov.shape[0]), seed=SIM_SEED)
    actual = sim.cov()

    assert covariance_frobenius_error(actual, expected) < SIM_TOL


def test_5_4_normal_simulation_higham_fix():
    cov = read_csv("test5_3.csv")
    expected = read_csv("testout_5.4.csv")

    fixed = higham_nearest_psd(cov)
    sim = simulate_normal(fixed, n_sim=100000, mean=np.zeros(cov.shape[0]), seed=SIM_SEED)
    actual = sim.cov()

    assert covariance_frobenius_error(actual, expected) < SIM_TOL


def test_5_5_pca_simulation_99_percent():
    cov = read_csv("test5_2.csv")
    expected = read_csv("testout_5.5.csv")

    sim = simulate_pca(
        cov,
        n_sim=100000,
        explained_variance=0.99,
        mean=np.zeros(cov.shape[0]),
        seed=SIM_SEED,
    )
    actual = sim.cov()

    assert covariance_frobenius_error(actual, expected) < SIM_TOL


# =========================================================
# Test 6: Returns
# =========================================================

def test_6_1_arithmetic_returns():
    data = read_csv("test6.csv")
    expected = read_csv("testout6_1.csv")

    actual = calculate_returns(data, method="arithmetic")

    assert actual["Date"].astype(str).tolist() == expected["Date"].astype(str).tolist()
    assert_matrix_close(
        actual.drop(columns=["Date"]),
        expected.drop(columns=["Date"]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_6_2_log_returns():
    data = read_csv("test6.csv")
    expected = read_csv("testout6_2.csv")

    actual = calculate_returns(data, method="log")

    assert actual["Date"].astype(str).tolist() == expected["Date"].astype(str).tolist()
    assert_matrix_close(
        actual.drop(columns=["Date"]),
        expected.drop(columns=["Date"]),
        atol=1e-12,
        rtol=1e-12,
    )


# =========================================================
# Test 7: Distribution fitting / regression
# =========================================================

def test_7_1_fit_normal():
    data = read_csv("test7_1.csv")
    expected = read_csv("testout7_1.csv").iloc[0]

    mu, sigma = fit_normal(data["x1"])

    assert np.isclose(mu, expected["mu"], atol=1e-10, rtol=1e-10)
    assert np.isclose(sigma, expected["sigma"], atol=1e-10, rtol=1e-10)


def test_7_2_fit_t():
    data = read_csv("test7_2.csv")
    expected = read_csv("testout7_2.csv").iloc[0]

    mu, sigma, nu = fit_t(data["x1"])

    assert np.isclose(mu, expected["mu"], atol=1e-6, rtol=1e-6)
    assert np.isclose(sigma, expected["sigma"], atol=1e-6, rtol=1e-6)
    assert np.isclose(nu, expected["nu"], atol=1e-5, rtol=1e-5)


def test_7_3_t_regression():
    data = read_csv("test7_3.csv")
    expected = read_csv("testout7_3.csv").iloc[0]

    X = data[["x1", "x2", "x3"]]
    y = data["y"]

    mu, sigma, nu, alpha, beta = fit_t_regression(X, y)

    actual = np.array([mu, sigma, nu, alpha, beta[0], beta[1], beta[2]])
    target = expected[["mu", "sigma", "nu", "Alpha", "B1", "B2", "B3"]].to_numpy(dtype=float)

    assert np.allclose(actual, target, atol=1e-5, rtol=1e-5)


def test_7_4_aicc_fitted_t():
    data = read_csv("test7_2.csv")
    expected = read_csv("testout7_4.csv").iloc[0]["AICC"]

    actual = calculate_aicc_t(data["x1"])
    assert np.isclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_7_5_nig_method_of_moments():
    data = read_csv("test7_5.csv")
    expected = read_csv("testout7_5.csv").iloc[0]

    actual = np.array(fit_nig_moments(data["x1"]))
    target = expected[["mu", "alpha", "beta", "delta"]].to_numpy(dtype=float)

    assert np.allclose(actual, target, atol=1e-5, rtol=1e-5)


def test_7_6_nig_mle():
    data = read_csv("test7_5.csv")
    expected = read_csv("testout7_6.csv").iloc[0]

    actual = np.array(fit_nig_mle(data["x1"]))
    target = expected[["mu", "alpha", "beta", "delta"]].to_numpy(dtype=float)

    assert np.allclose(actual, target, atol=1e-4, rtol=1e-4)
