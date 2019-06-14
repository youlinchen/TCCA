# Tensor Canonical Correlation Analysis

The code that accompanies the paper "Tensor Canonical Correlation Analysis"
by Chen, Kolar and Tsay.   
https://arxiv.org/abs/1906.05358


## Simulation
Figures 1 and 2 in the TCCA paper are generated by `Figure_1.ipynb` and `Figure_2.ipynb`, respectively.

## Gene Expression Data
Run the code available at https://github.com/pachterlab/PCCA/ first.
The results are then generated by `geno_expression_gTCCA.ipynb` and `geno_expression_gTCCA_deflation.ipynb`.

## Air Pollution Data in Taiwan
Original data can be download from https://taqm.epa.gov.tw/taqm/en/YearlyDataDownload.aspx.
The result are generated by `twn_air_pollution.ipynb`.

Note that you will need to install the package `folium` for visualization.  
If you are using `conda` for managing packages, you can run  

    conda install -c conda-forge folium

to install the package.

## Electricity Demands in Adelaide
Data are from the R package `fds`. See https://cran.r-project.org/web/packages/fds/index.html for more details.
The result are generated by `demand_vs_temp.ipynb`.
