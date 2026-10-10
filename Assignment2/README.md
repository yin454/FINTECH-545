# FINTECH 545 - Assignment 2

## Reproduce the results

From the `Assignment2` folder, run:

```bash
python assignment2.py
```

The script prints all numerical results to the terminal and writes the figures to `plots/`.

To save the printed output to a file:

```bash
python assignment2.py > results.txt
```

## Conventions

- Unless stated otherwise, VaR and ES use `alpha = 5%` and are reported with the positive-loss sign convention.
- Arithmetic returns are used when the assignment asks for returns.
- Problem 2 removes the full-sample return mean before calculating volatility and VaR.
- Problem 4 uses ranks `r/(n+1)` for the preliminary rank plots and fitted marginal CDFs for copula estimation.
- For the copula comparison, the fitted margins and the common Kendall-tau correlation matrix are frozen, following Week 5 Section 6.5. Therefore Gaussian uses `k=0` and the t copula uses `k=1` for `nu`.
- Problem 5 follows the instruction to assume zero expected returns in simulation, so fitted intercepts are reported but are not added to simulated returns.
- Random seeds are fixed in `assignment2.py` so simulation results are reproducible.
