# Tensor Canonical Correlation Analysis

The code that accompanies the paper "Tensor Canonical Correlation Analysis withConvergence and Statistical Guarantees"
by Chen, Kolar and Tsay.   
https://arxiv.org/abs/1906.05358

# Basic Usage

We implement several algorithms in util.py

-  ``twoDcca(X, Y)`` – 2DCCA for first left and right canonical component.
-  ``twoDcca_mat(X, Y, p1, p2)`` - (``p1``,``p2``) - 2DCCA or p1 left and p3 right canonical components.
-  ``twoDcca_deflation(X, Y)`` - The deflation procedure of two 2DCCA components which is only implement for inexact updating

Please see `example_util.ipynb` for more details.

## comparison
We compute TCCA for three datasets in `Comparision.ipynb`. The result is shown in Table 1.

Reference:
AppGrad: https://github.com/MaZhuang/CCA 

rifle: https://cran.r-project.org/web/packages/rifle/index.html

## Air Pollution Data in Taiwan
Original data can be download from https://taqm.epa.gov.tw/taqm/en/YearlyDataDownload.aspx.
We produce the results in section 6.1.2 in `twn_air_pollution.ipynb`.
