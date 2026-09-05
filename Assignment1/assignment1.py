import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

from risklib import (
    sample_moments,
    fit_normal_regression,
    fit_student_t_regression,
    conditional_normal_parameters,
    fit_ar_ma_models,
)


def problem1():
    print("\n" + "=" * 60)
    print("PROBLEM 1")
    print("=" * 60)

    df = pd.read_csv("problem1.csv")
    x = df["x"].to_numpy()
    n = len(x)

    moments = sample_moments(x)

    mean = moments["mean"]
    variance = moments["variance"]
    std_dev = np.sqrt(variance)

    # Fit the Normal by matching the sample mean and variance.
    normal_q01 = stats.norm.ppf(
        0.01,
        loc=mean,
        scale=std_dev
    )

    actual_below = np.sum(x < normal_q01)
    expected_below = 0.01 * n

    print("Mean:", mean)
    print("Variance:", variance)
    print("Skewness:", moments["skewness"])
    print("Excess kurtosis:", moments["excess_kurtosis"])
    print("Fitted Normal standard deviation:", std_dev)
    print("Fitted Normal 1% quantile:", normal_q01)
    print("Actual observations below 1% quantile:", actual_below)
    print("Expected observations below 1% quantile:", expected_below)


def problem2():
    print("\n" + "=" * 60)
    print("PROBLEM 2")
    print("=" * 60)

    df = pd.read_csv("problem2.csv")
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # Plot before fitting, as required by the assignment.
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Problem 2: y versus x")
    plt.tight_layout()
    plt.savefig("problem2_scatter.png", dpi=200)
    plt.close()

    # OLS with standard errors.
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()

    normal_fit = fit_normal_regression(x, y)
    t_fit = fit_student_t_regression(x, y)

    print("OLS alpha:", ols.params[0])
    print("OLS beta:", ols.params[1])
    print("OLS SE(alpha):", ols.bse[0])
    print("OLS SE(beta):", ols.bse[1])
    print("OLS residual standard error:", np.sqrt(ols.scale))

    print("\nNormal-error MLE")
    print("alpha:", normal_fit["alpha"])
    print("beta:", normal_fit["beta"])
    print("sigma:", normal_fit["sigma"])
    print("AICc:", normal_fit["aicc"])

    print("\nStudent-t-error MLE")
    print("alpha:", t_fit["alpha"])
    print("beta:", t_fit["beta"])
    print("sigma (scale):", t_fit["sigma"])
    print("nu:", t_fit["nu"])
    print("AICc:", t_fit["aicc"])

    print("\nError quantiles")

    for probability in [0.95, 0.995]:
        normal_quantile = stats.norm.ppf(
            probability,
            loc=0,
            scale=normal_fit["sigma"]
        )

        t_quantile = stats.t.ppf(
            probability,
            df=t_fit["nu"],
            loc=0,
            scale=t_fit["sigma"]
        )

        print(
            f"{probability:.3f}: "
            f"Normal = {normal_quantile:.6f}, "
            f"Student-t = {t_quantile:.6f}"
        )


def problem3():
    print("\n" + "=" * 60)
    print("PROBLEM 3")
    print("=" * 60)

    df = pd.read_csv("problem3.csv")

    columns = ["x1", "x2", "x3", "x4"]

    # Plot every pair before computing correlations.
    pd.plotting.scatter_matrix(
        df[columns],
        figsize=(9, 9),
        diagonal="hist",
        alpha=0.65
    )

    plt.suptitle(
        "Problem 3: Pairwise Scatter Plots",
        y=0.92
    )
    plt.tight_layout()
    plt.savefig("problem3_pairplot.png", dpi=200)
    plt.close()

    pearson = df[columns].corr(method="pearson")
    spearman = df[columns].corr(method="spearman")

    gap = (pearson - spearman).abs()

    # Ignore the diagonal when looking for the largest gap.
    upper_triangle = np.triu(
        np.ones(gap.shape, dtype=bool),
        k=1
    )

    gap_values = gap.to_numpy().copy()
    gap_values[~upper_triangle] = -np.inf

    i, j = np.unravel_index(
        np.argmax(gap_values),
        gap_values.shape
    )

    print("Pearson correlation matrix:")
    print(pearson)

    print("\nSpearman correlation matrix:")
    print(spearman)

    print("\nLargest Pearson-Spearman gap:")
    print(columns[i], columns[j], gap.iloc[i, j])


def problem4():
    print("\n" + "=" * 60)
    print("PROBLEM 4")
    print("=" * 60)

    df = pd.read_csv("problem4.csv")
    x1 = df["x1"].to_numpy()
    x2 = df["x2"].to_numpy()

    result = conditional_normal_parameters(x1, x2)

    mu1 = result["mu1"]
    mu2 = result["mu2"]
    beta = result["beta"]
    conditional_variance = result["conditional_variance"]

    conditional_mean = (
        mu2
        + beta * (x1 - mu1)
    )

    # Under multivariate Normality the conditional variance is constant.
    half_width = (
        stats.norm.ppf(0.975)
        * np.sqrt(conditional_variance)
    )

    lower = conditional_mean - half_width
    upper = conditional_mean + half_width

    inside = (
        (x2 >= lower)
        & (x2 <= upper)
    )

    overall_coverage = inside.mean()

    # Split observations by the distance of x1 from its sample mean.
    x1_std = np.std(x1, ddof=1)

    distance_in_sd = np.abs(
        (x1 - mu1) / x1_std
    )

    bucket1 = distance_in_sd <= 1
    bucket2 = (
        (distance_in_sd > 1)
        & (distance_in_sd <= 2)
    )
    bucket3 = distance_in_sd > 2

    # Sort x1 only for plotting the conditional mean and bands.
    order = np.argsort(x1)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        x1,
        x2,
        alpha=0.45,
        label="Data"
    )
    plt.plot(
        x1[order],
        conditional_mean[order],
        label="Conditional mean"
    )
    plt.plot(
        x1[order],
        lower[order],
        linestyle="--",
        label="95% band"
    )
    plt.plot(
        x1[order],
        upper[order],
        linestyle="--"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(
        "Problem 4: Conditional Mean with 95% Band"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "problem4_conditional_band.png",
        dpi=200
    )
    plt.close()

    print("Sample covariance matrix:")
    print(result["covariance"])

    print("Conditional mean coefficient beta:", beta)
    print("Conditional variance:", conditional_variance)
    print(
        "Remaining variance factor:",
        result["variance_factor"]
    )
    print("95% band half-width:", half_width)
    print("Overall coverage:", overall_coverage)

    print(
        "Coverage within 1 SD:",
        inside[bucket1].mean(),
        "n =", bucket1.sum()
    )

    print(
        "Coverage between 1 and 2 SD:",
        inside[bucket2].mean(),
        "n =", bucket2.sum()
    )

    print(
        "Coverage beyond 2 SD:",
        inside[bucket3].mean(),
        "n =", bucket3.sum()
    )


def problem5():
    print("\n" + "=" * 60)
    print("PROBLEM 5")
    print("=" * 60)

    df = pd.read_csv("problem5.csv")
    x = df["x"].to_numpy()
    n = len(x)

    significance_band = 1.96 / np.sqrt(n)

    # Time-series plot.
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(n), x)
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("Problem 5: Time Series")
    plt.tight_layout()
    plt.savefig("problem5_series.png", dpi=200)
    plt.close()

    # ACF.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    sm.graphics.tsa.plot_acf(
        x,
        lags=20,
        alpha=0.05,
        ax=ax
    )
    ax.set_title("Problem 5: ACF")
    plt.tight_layout()
    plt.savefig("problem5_acf.png", dpi=200)
    plt.close()

    # PACF.
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    sm.graphics.tsa.plot_pacf(
        x,
        lags=20,
        alpha=0.05,
        method="ywmle",
        ax=ax
    )
    ax.set_title("Problem 5: PACF")
    plt.tight_layout()
    plt.savefig("problem5_pacf.png", dpi=200)
    plt.close()

    print(
        "Approximate 95% significance band:",
        (-significance_band, significance_band)
    )

    fits = fit_ar_ma_models(x)

    print("\nAICc values")

    for name, model in fits.items():
        print(
            name,
            "AICc =",
            model.aicc
        )

    print("\nAR(2) parameters")
    print(fits["AR(2)"].params)

    print("\nAR(3) parameters")
    print(fits["AR(3)"].params)


if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
