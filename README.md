# Bias-Correction-QQ

**Bias Correction using Q–Q Mapping / CDF Matching (Gamma Distribution)**  
This Python module applies bias correction to climate model data (historical and future) using **quantile-to-quantile (Q–Q) mapping** via a parametric Gamma distribution, following the methodology of *Salvi et al., 2013*.

It supports region-aware seasonal definitions and works efficiently with gridded datasets in NetCDF format using `xarray`.

---

## Features

- **Q–Q Mapping** (CDF Matching) using Gamma distributions
- **Region-aware seasons** for India, US, Australia, or custom
- Modular and reusable design for temperature/precipitation correction
- Option to save corrected NetCDF outputs
- Based on parametric bias correction (not empirical quantile mapping)

---

## Method

This approach performs:
- Gamma distribution fitting for observed, historical model, and future model data
- CDF matching between model and observed data (quantile-to-quantile)
- Optional adjustment of future model output using a shift derived from modeled and observed distributions

---

## Usage

```python
from bias_correction_gamma_salvi_modular import run_bias_correction

hist_corr, fut_corr = run_bias_correction(
    hist, fut, obs,
    save_path="./output",    # optional folder to save NetCDF files
    region="India"           # "India", "US", "Australia", or leave blank for default seasons
)
