# FinTech 545 — Assignment 1

## Run the assignment

Open a terminal in the `Assignment1` folder and run:

```bash
python assignment1.py
```

The script runs Problems 1 through 5 in order and prints all numerical results used in the written response.

It also creates the required figures:

```text
problem2_scatter.png
problem3_pairplot.png
problem4_conditional_band.png
problem5_series.png
problem5_acf.png
problem5_pacf.png
```

## Code organization

`risklib.py` contains reusable statistical and modeling functions.

`assignment1.py` reads the five CSV files, calls the functions in `risklib.py`, prints the results, and creates the figures.

This structure keeps the reusable library code separate from the assignment-specific analysis.

## Conventions used

### Moments

- Sample variance uses `ddof=1`.
- Skewness uses the bias-corrected sample estimator.
- Kurtosis is reported as **excess kurtosis**, so a Normal distribution has excess kurtosis equal to 0.

### Regression

The Normal-error and Student-t-error regressions are estimated by maximum likelihood.

For the Student-t model:

- `sigma` is the scale parameter.
- `nu` is the fitted degrees of freedom.

### AICc

AICc is calculated as:

```text
AICc = AIC + 2*k*(k+1)/(n-k-1)
```

where:

- `k` is the number of fitted parameters.
- `n` is the number of observations.

### Conditional distribution

The conditional model in Problem 4 uses the sample covariance matrix and the multivariate Normal conditioning formulas.

### AR and MA models

AR(1) through AR(3) and MA(1) through MA(3) include a constant and are fit using `statsmodels.tsa.arima.model.ARIMA`.
