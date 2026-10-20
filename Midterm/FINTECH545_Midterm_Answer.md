# FINTECH 545 — Midterm Exam

**Name:** [Leah Yin]  
**Date:** [10/20/2026]

---

## Problem 1 — [Pairwise Correlation with Missing Data]

### (a) [Calculate the pairwise correlation matrix.]

```python
          x1        x2        x3        x4        x5
x1  1.000000  0.332836  0.418462  0.255285  0.732272
x2  0.332836  1.000000  0.251035  0.287709  0.787538
x3  0.418462  0.251035  1.000000  0.483580  0.684257
x4  0.255285  0.287709  0.483580  1.000000  0.782632
x5  0.732272  0.787538  0.684257  0.782632  1.000000
```

### (b) [Calculate the variance of each series, using every observation that series has.]
```python
x1    0.000406
x2    0.000480
x3    0.000151
x4    0.000842
x5    0.000273
```

### (c) [Is the correlation matrix positive semi-definite? Show how you know.]
Eigenvalues of Original Correlation Matrix: [-0.08927867  0.4613398   0.71908517  0.82343722  3.08541647]. There is a negative number -0.08927867, so the correlation matrix is not positive semi-definite.

### (d) [If it is not, use Higham’s method to find the nearest positive semi-definite correlation matrix. Report the matrix and its eigenvalues. Which entries changed the most?]
```python
          x1        x2        x3        x4        x5
x1  1.000000  0.349452  0.425942  0.270782  0.694572
x2  0.349452  1.000000  0.260599  0.307523  0.739337
x3  0.425942  0.260599  1.000000  0.492500  0.662558
x4  0.270782  0.307523  0.492500  1.000000  0.737677
x5  0.694572  0.739337  0.662558  0.737677  1.000000
```
Eigenvalues of Repaired Matrix: [-5.29177268e-10  4.50081533e-01  7.03476106e-01  8.06708468e-01  3.03973389e+00]. Pair x2-x5 changed the most.

### (e) [Combine your results into a covariance matrix. Report it.]
```python
          x1        x2        x3        x4        x5
x1  0.000406  0.000154  0.000106  0.000158  0.000231
x2  0.000154  0.000480  0.000070  0.000195  0.000268
x3  0.000106  0.000070  0.000151  0.000176  0.000135
x4  0.000158  0.000195  0.000176  0.000842  0.000354
x5  0.000231  0.000268  0.000135  0.000354  0.000273
```
---

## Problem 2 — [VaR and ES for a Single Stock]

### (a) [What is your VaR at the 95% and 99% levels, in percent? Explain your method.]

| Confidence Level   |   VaR (%) |   VaR ($) |
|:-------------------|----------:|----------:|
| 95%                |    3.2213 |   240.043 |
| 99%                |    4.5559 |   339.498 |

First, I fit a normal regression for r_XYZ = alpha + beta * r_SPY + epsilon, and get the alpha, beta and residuals. Then, fit a normal distribution for r_spy and residuals seperatly, and get their mean and standard deviation. Because we assume the arithmetic returns of SPY and epsilon are normally distributed, the r_XYZ is also normally distributed. We can get the mean and standard deviation of XYZ through SPY and epsilon. Then get the normal VaR at the 95% and 99% levels.

### (b) [What is your ES at the 95% and 99% levels, in percent? Explain your method.]

| Confidence Level   |   ES (%) |   ES ($) |
|:-------------------|---------:|---------:|
| 95%                |   4.0396 |  301.024 |
| 99%                |   5.2195 |  388.951 |

Follow the steps in (a), we can get the normal ES at the 95% and 99% levels.

---

## Problem 3 — [Copula VaR and ES for a Two Book Portfolio]

### (a) [For each stock, fit a normal, a Student’s 𝑡, and a normal inverse Gaussian distribution to its returns. Report the fitted parameters. Which distribution fits each stock best? Justify your choice.]
| Stock   | Distribution   |      mu |    sigma |               nu |     alpha |      beta |    delta |   Log Likelihood |     AICc |
|:--------|:---------------|--------:|---------:|-----------------:|----------:|----------:|---------:|-----------------:|---------:|
| S1      | normal         |  0.0009 |   0.0118 |    nan           |  nan      |  nan      | nan      |          3019    | -6033.99 |
| S1      | t              |  0.0009 |   0.0118 |      3.55369e+08 |  nan      |  nan      | nan      |          3019    | -6031.98 |
| S1      | NIG            |  0.0766 | nan      |    nan           | 2086.49   | -588.204  |   0.2575 |          3019.13 | -6030.22 |
| S2      | normal         |  0.0008 |   0.0153 |    nan           |  nan      |  nan      | nan      |          2758.36 | -5512.71 |
| S2      | t              |  0.0006 |   0.0117 |      4.8579      |  nan      |  nan      | nan      |          2816.69 | -5627.36 |
| S2      | NIG            | -0.0003 | nan      |    nan           |   73.2066 |    4.8638 |   0.0166 |          2815.05 | -5622.06 |
| S3      | normal         |  0.0004 |   0.0189 |    nan           |  nan      |  nan      | nan      |          2550.26 | -5096.51 |
| S3      | t              |  0.0023 |   0.0137 |      4.1605      |  nan      |  nan      | nan      |          2622.24 | -5238.46 |
| S3      | NIG            |  0.0105 | nan      |    nan           |   79.8367 |  -36.3446 |   0.0197 |          2644.84 | -5281.64 |
| S4      | normal         |  0.0008 |   0.0099 |    nan           |  nan      |  nan      | nan      |          3193.49 | -6382.97 |
| S4      | t              |  0.0008 |   0.0099 |      3.39036e+08 |  nan      |  nan      | nan      |          3193.49 | -6380.96 |
| S4      | NIG            |  0.0895 | nan      |    nan           | 3368.19   | -984.958  |   0.2903 |          3193.58 | -6379.12 |
| S5      | normal         |  0.0007 |   0.0189 |    nan           |  nan      |  nan      | nan      |          2547.45 | -5090.9  |
| S5      | t              |  0.0008 |   0.0156 |      6.2892      |  nan      |  nan      | nan      |          2578.45 | -5150.87 |
| S5      | NIG            |  0.0015 | nan      |    nan           |   73.5768 |   -2.2326 |   0.026  |          2577.12 | -5146.21 |
| S6      | normal         |  0.0008 |   0.0151 |    nan           |  nan      |  nan      | nan      |          2775.61 | -5547.2  |
| S6      | t              |  0.0024 |   0.0107 |      3.8238      |  nan      |  nan      | nan      |          2847.38 | -5688.74 |
| S6      | NIG            |  0.0083 | nan      |    nan           |   91.6825 |  -41.6862 |   0.0146 |          2872.72 | -5737.4  |
| S7      | normal         |  0.0007 |   0.0137 |    nan           |  nan      |  nan      | nan      |          2873.85 | -5743.69 |
| S7      | t              |  0.0007 |   0.0137 | 319103           |  nan      |  nan      | nan      |          2873.85 | -5741.68 |
| S7      | NIG            |  0.0775 | nan      |    nan           | 1635.96   | -443.713  |   0.2726 |          2873.96 | -5739.89 |
| S8      | normal         |  0.0004 |   0.0171 |    nan           |  nan      |  nan      | nan      |          2649.56 | -5295.11 |
| S8      | t              |  0.0007 |   0.0114 |      3.4833      |  nan      |  nan      | nan      |          2749.66 | -5493.29 |
| S8      | NIG            |  0.0012 | nan      |    nan           |   50.6228 |   -2.8868 |   0.0142 |          2747.57 | -5487.09 |
| S9      | normal         |  0.0017 |   0.0206 |    nan           |  nan      |  nan      | nan      |          2461.76 | -4919.51 |
| S9      | t              |  0.0034 |   0.0153 |      4.3852      |  nan      |  nan      | nan      |          2521.39 | -5036.76 |
| S9      | NIG            |  0.0101 | nan      |    nan           |   63.1595 |  -22.9267 |   0.0215 |          2536.77 | -5065.51 |
| S10     | normal         |  0.0002 |   0.0131 |    nan           |  nan      |  nan      | nan      |          2919.67 | -5835.32 |
| S10     | t              |  0.0004 |   0.0094 |      4.1213      |  nan      |  nan      | nan      |          2990.44 | -5974.86 |
| S10     | NIG            |  0.001  | nan      |    nan           |   76.6857 |   -5.3684 |   0.0126 |          2989.37 | -5970.7  |

The best distribution has the smallest AICc:

**Distribution Selection**
| Stock   |   Normal AICc |   Student-t AICc |   NIG AICc | Selected   |
|:--------|--------------:|-----------------:|-----------:|:-----------|
| S1      |      -6033.99 |         -6031.98 |   -6030.22 | normal     |
| S2      |      -5512.71 |         -5627.36 |   -5622.06 | t          |
| S3      |      -5096.51 |         -5238.46 |   -5281.64 | NIG        |
| S4      |      -6382.97 |         -6380.96 |   -6379.12 | normal     |
| S5      |      -5090.9  |         -5150.87 |   -5146.21 | t          |
| S6      |      -5547.2  |         -5688.74 |   -5737.4  | NIG        |
| S7      |      -5743.69 |         -5741.68 |   -5739.89 | normal     |
| S8      |      -5295.11 |         -5493.29 |   -5487.09 | t          |
| S9      |      -4919.51 |         -5036.76 |   -5065.51 | NIG        |
| S10     |      -5835.32 |         -5974.86 |   -5970.7  | t          |

### (b) [Using the distributions you chose in part (a), fit a Gaussian copula and a Student’s t copula. Which one fits better? Justify your choice.]
(The Gaussian then has no free parameters left and the t has one, so here we set k_Gauss = 0 and k_𝑡 = 1)
| Copula    |   nu |   Log Likelihood |     AICc |
|:----------|-----:|-----------------:|---------:|
| Gaussian  |  nan |          2110.03 | -4220.05 |
| Student-t |  100 |          2108.16 | -4214.32 |

Selected Copula: Gaussian, whose AICc is smaller.

### (c) [Using the copula you chose in part (b), simulate the portfolio with at least 5,000 draws. Calculate the absolute VaR and ES at the 95% level, in dollars, for each stock, each book, and the total portfolio. Present the results in a table. Discuss the results.]
| Position        |     VaR |       ES |
|:----------------|--------:|---------:|
| S1              |  901.1  |  1129.94 |
| S2              | 1064.78 |  1530.31 |
| S3              |  796.08 |  1266.6  |
| S4              |  671.14 |   845.5  |
| S5              | 1086.08 |  1517.02 |
| S6              | 1219.26 |  1901.03 |
| S7              |  880    |  1109.62 |
| S8              |  646.83 |  1019.46 |
| S9              | 3414.82 |  5367.17 |
| S10             |  474.77 |   747.65 |
| Book A          | 3583.38 |  4830.6  |
| Book B          | 5418.72 |  8195.99 |
| Total Portfolio | 7765.91 | 10909.3  |

| Position        |   Market Value ($) |
|:----------------|-------------------:|
| Book A          |             200618 |
| Book B          |             235278 |
| Total Portfolio |             435896 |

Book B has larger VaR and ES at the 95% level, because the stock in Book B has larger VaR and ES on average than Book A. But Book B has larger market value than Book A. The result shows that high risk usually accompany with high income at some level.