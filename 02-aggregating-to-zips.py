# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %load_ext rpy2.ipython
# %matplotlib inline

# %%
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
from IPython.display import clear_output
from sklearn.metrics.pairwise import haversine_distances
from statsmodels.formula import api as smf

from util.data import (
    generate_zillow_data,
    get_census_shapefiles,
    get_census_tract_attributes,
    get_oz_data,
    get_pairs,
    get_zips,
    get_file_suffix,
)
from util.plot import plot_with_error_bars

try:
    import pandas_tools.latex as tex
except ImportError:
    print("Run pip install git+https://github.com/jiafengkevinchen/pandas_tools")

try:
    from janitor.utils import skiperror, skipna
except ImportError:
    try:
        from pandas_tools.latex import skiperror, skipna
    except ImportError:
        print("Run pip install git+https://github.com/jiafengkevinchen/pandas_tools")

# %%
with open("settings.json", "r") as f:
    settings = json.load(f)

LIC_ONLY = settings["LIC_ONLY"]
OVERWRITE = settings["OVERWRITE"]
START_YEAR = settings["START_YEAR"]
file_suffix = get_file_suffix(LIC_ONLY)

table = pd.read_pickle(f"exhibits/table_script1{file_suffix}.pickle")
df, annual_change, oz_irs, oz_ui = get_oz_data()
tracts_df, var_dict = get_census_tract_attributes()

zip_panel = get_zips(start_year=START_YEAR, overwrite=OVERWRITE, lic_only=LIC_ONLY)

# %%
with open(f"data/zip_missing{file_suffix}.txt", "r") as f:
    n_selected_tracts_covered, non_missing_zips, total_zips = [
        float(s) for s in f.read().split()
    ]

# %%
with open(f"exhibits/zip_coverage{file_suffix}.tex", "w") as f:
    print(
        f"""\
        Although only {tex.mathify(non_missing_zips)} of
        the total {tex.mathify(total_zips)} ZIP codes with crosswalk data
        do not have missing data in 2018,
        these ZIP codes intersect with
        {tex.mathify(n_selected_tracts_covered)} selected Opportunity Zones.% """,
        file=f,
    )

# %%
zip_panel = zip_panel.query("status_eligible_not_selected + status_selected > 0")

# %%
coverage = (
    zip_panel.query("year == 2018")[["zip_code", "status_selected"]]
    .drop_duplicates()
    .describe()
)

# %%
mean_coverage = coverage.loc["mean"].iloc[0]
median_coverage = coverage.loc["50%"].iloc[0]
coverage75 = coverage.loc["75%"].iloc[0]

with open(f"exhibits/zip_coverage_distro{file_suffix}.tex", "w") as f:
    print(
        f"""\
        The average ZIP code has {mean_coverage*100:.1f}\% of
        its addresses in a selected Opportunity Zone;
        the median ZIP code has {median_coverage*100:.1f}\%;
        and the 75th percentile has {coverage75*100:.1f}\%.%""",
        file=f,
    )

# %% [markdown]
# # Zillow design

# %%
# %Rpush zip_panel

# %% {"language": "R"}
# source("util/twfe_did.r")

# %% [markdown]
# ## TWFE

# %% {"language": "R"}
# zip_panel$post <- zip_panel$year == "2018"
# zip_panel$zip_code <- factor(zip_panel$zip_code)
# zip_panel$year <- relevel(factor(zip_panel$year), ref = "2017")
# zip_panel$treatment <- zip_panel$status_selected
#
# model_pretest_zip <- fit_did(
#   fmla = annual_change ~ 1 + year + treatment * post,
#   pretest_fmla = annual_change ~ 1 + year * treatment,
#   data = zip_panel,
#   pretest_cols = c(
#     "year2014:treatment=0",
#     "year2015:treatment=0",
#     "year2016:treatment=0"
#   ),
#   index = c("zip_code", "year")
# )
#
# model_pretest_zip_covs <- fit_did(
#   pretest_fmla = annual_change ~ 1 + year * treatment + treatment * post + year * (
#     log_median_household_income + total_housing + pct_white + pct_higher_ed + pct_rent +
#       pct_native_hc_covered + pct_poverty + pct_supplemental_income + pct_employed),
#   fmla = annual_change ~ 1 + treatment * post + year * (log_median_household_income +
#     total_housing + pct_white + pct_higher_ed + pct_rent +
#     pct_native_hc_covered + pct_poverty + pct_supplemental_income + pct_employed),
#   pretest_cols = c(
#     "year2014:treatment=0",
#     "year2015:treatment=0",
#     "year2016:treatment=0"
#   ),
#   data = zip_panel,
#   index = c("zip_code", "year")
# )

# %%
model_pretest_zip = %Rget model_pretest_zip
model_pretest_zip_covs = %Rget model_pretest_zip_covs

# %%
tau, se, _, pval = np.array(model_pretest_zip.rx["coeftest_model"][0])[-1, :]
tau_cov, se_cov, _, pval_cov = np.array(model_pretest_zip_covs.rx["coeftest_model"][0])[
    4, :
]
pretest_zip_pval = np.array(model_pretest_zip.rx["lh_pretest"][0])[-1, -1]
pretest_zip_cov_pval = np.array(model_pretest_zip_covs.rx["lh_pretest"][0])[-1, -1]
n = zip_panel["zip_code"].nunique()

# %%
table_zip = pd.DataFrame(
    {
        "TWFE": [
            tau,
            f"({se})",
            pval,
            pretest_zip_pval,
            n,
            "No",
            f"Unbalanced ({START_YEAR}--2018)",
        ],
        "TWFE ": [
            tau_cov,
            f"({se_cov})",
            pval_cov,
            pretest_zip_cov_pval,
            n,
            "Yes",
            f"Unbalanced ({START_YEAR}--2018)",
        ],
    },
    index=[
        r"$\hat \tau$",
        "",
        r"$p$-value",
        "Pre-trend test $p$-value",
        "$N$",
        "Covariates",
        "Sample",
    ],
)

# %% [markdown]
# ## TWFE variable selection

# %% {"language": "R"}
# model_pretest_zip_cov_test <- fit_did(
#   pretest_fmla = annual_change ~ 1 + year * treatment + treatment * post +
#     year * (log_median_household_income + pct_white),
#   fmla = annual_change ~ 1 + treatment * post +
#     year * (log_median_household_income + pct_white),
#   pretest_cols = c(
#     "year2014:treatment=0",
#     "year2015:treatment=0",
#     "year2016:treatment=0"
#   ),
#   data = zip_panel,
#   index = c("zip_code", "year")
# )

# %%
model_pretest_zip_cov_test = %R model_pretest_zip_cov_test

# %%
with open("exhibits/script1_data.txt", "r") as f:
    tau_sp, se_sp = map(float, f.read().split())
t_spz, se_spz, _, _ = np.array(model_pretest_zip_cov_test.rx["coeftest_model"][0])[4, :]

# %%
covariate_choice = f"""\
For Column (2), only including log median household income and
percent white as covariates gives {tex.mathify(tau_sp)} ({tex.mathify(se_sp)}) for the top panel and
{tex.mathify(t_spz)} ({tex.mathify(se_spz)}) for the bottom panel.
"""

# %% [markdown]
# ## Weighting

# %%
covs = [
    "log_median_household_income_notnull",
    "total_housing_notnull",
    "pct_white_notnull",
    "pct_higher_ed_notnull",
    "pct_rent_notnull",
    "pct_native_hc_covered_notnull",
    "pct_poverty_notnull",
    "pct_supplemental_income_notnull",
    "pct_employed_notnull",
]

pct_treated = oz_ui.query("designated == 'Selected'").count().iloc[0] / len(tracts_df)

two_period_zip = zip_panel.query("year == 2017 or year == 2018").set_index("zip_code")
pre = two_period_zip.query("year == 2017").sort_values("zip_code")
post = (
    two_period_zip.query("year == 2018")
    .assign(
        treatment=lambda x: x.status_selected
        >= x.status_selected.quantile(1 - pct_treated)
    )
    .sort_values("zip_code")
)
covar = (
    two_period_zip[covs]
    .reset_index()
    .drop_duplicates()
    .sort_values("zip_code")
    .drop("zip_code", axis=1)
)

null = (
    pre[["annual_change"]].isnull().any(axis=1).values
    | post[["annual_change"]].isnull().any(axis=1).values
    | covar.isnull().any(axis=1).values
)
pre = pre[~null].copy()
post = post[~null].copy()
covar = covar[~null].copy()

# %%
# %Rpush pre post covar

# %% {"language": "R"}
# library(DRDID)
# pre <- as.data.frame(pre)
# post <- as.data.frame(post)
# covariates <- as.data.frame(covar)
#
# drdid_out <- drdid_imp_panel(
#     y1 = post$annual_change,
#     y0 = pre$annual_change,
#     D = post$treatment,
#     covariates = covariates,
# )
#
# drdid_out

# %%
drdid_out = %Rget drdid_out

# %%
drdid_att = drdid_out.rx["ATT"][0][0]
drdid_se = drdid_out.rx["se"][0][0]

# %%
table_zip["Weighting DR"] = [
    drdid_att,
    f"({drdid_se})",
    (2 * (1 - scipy.stats.norm.cdf(abs(drdid_att / drdid_se)))),
    None,
    (post["treatment"].sum(), (1 - post["treatment"]).sum()),
    "Yes",
    "Balanced (2017--2018)",
]
table_zip.loc["Model", :] = ["Within", "Within", "Weighting"]

# %%
if "TWFE" not in table.columns:
    table.columns = [
        "TWFE",
        "TWFE ",
        "Weighting CS",
        "Weighting DR",
        "Paired",
        "Paired ",
    ]

_ = (
    pd.concat(
        [
            pd.DataFrame(
                {c: "" for c in table.columns}, index=["\\textbf{Tract-level data}"]
            ),
            table.fillna("---").rename(index={c: f"\\quad {c}" for c in table.index}),
            pd.DataFrame(
                {c: "" for c in table.columns},
                index=["\\midrule \\textbf{ZIP-level data}"],
            ),
            table_zip.fillna("---").rename(
                index={c: f"\\quad {c}" for c in table_zip.index}
            ),
        ],
        sort=False,
    )
    .fillna("")
    .to_latex_table(
        caption="Estimation of ATT using FHFA Tract and ZIP-level data",
        label=f"tract_and_zip{file_suffix}",
        additional_text="\\scriptsize",
        notes=f"""\
    Covariates include log median household income, total housing units, percent white,
    percent with post-secondary education,
    percent rental units, percent covered by health insurance among native-born individuals,
    percent below poverty line, percent receiving supplemental income, and percent employed.
    {covariate_choice}
    Pretest for Column (2) interacts covariates with time dummies.
    Discrete treatment in Column (4) is defined as
    the highest {(1 - pct_treated) * 100:.1f}\% of treated
    tract coverage, so as to keep the percentage of treated ZIPs the same as treated tracts.
    """,
        filename=f"exhibits/join_tab{file_suffix}.tex",
    )
)


# %% [markdown]
# # Redoing ZIP code TWFE by splitting on covariate

# %%
def get_emp_pct():
    population = pd.read_csv("data/ACS_16_5YR_DP05/ACS_16_5YR_DP05_with_ann.csv").iloc[
        1:, [1, 3]
    ]
    population.columns = ["zip_code", "population"]
    population["population"] = population["population"].astype(float)
    population.loc[population["population"] < 1, "population"] = np.nan

    emp_pct = (
        pd.read_csv("data/zbp16totals.txt")[["zip", "emp"]]
        .assign(zip_code=lambda x: x.zip.astype(str).str.zfill(5))
        .merge(
            population.assign(zip_code=lambda x: x.zip_code.astype(str).str.zfill(5)),
            how="outer",
        )
        .drop("zip", axis=1)
        .assign(emp_pct=lambda x: x.emp / x.population)
    )[["zip_code", "emp_pct"]]
    return emp_pct


# %%
try:
    emp_pct = get_emp_pct()
except FileNotFoundError:
    !wget -q -O data/cbp.zip https://www2.census.gov/programs-surveys/cbp/datasets/2016/zbp16totals.zip
    !unzip -q data/cbp.zip -d data/
    emp_pct = get_emp_pct()

# %%
quartile_thresh = (
    zip_panel.merge(emp_pct)[["zip_code", "emp_pct"]]
    .drop_duplicates()["emp_pct"]
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .values
)
by_quartile = []
for i, q in enumerate(quartile_thresh):
    if i == 0:
        continue
    qprev = quartile_thresh[i - 1]
    data = zip_panel.merge(emp_pct).query("@qprev < emp_pct <= @q")
    by_quartile.append(data)

# %%
with open(f"exhibits/quartiles{file_suffix}.tex", "w") as f:
    print(
        f"""\
    The quartile thresholds are {', '.join(map(tex.mathify, quartile_thresh[1:-1]))} in the employment to population ratio. % """,
        file=f,
    )

# %%
q1, q2, q3, q4 = by_quartile
# %Rpush q1 q2 q3 q4

# %% {"language": "R"}
# model_lst <- list()
#
# for (data in list(q1, q2, q3, q4)) {
#   data$post <- data$year == "2018"
#   data$zip_code <- factor(data$zip_code)
#   data$year <- relevel(factor(data$year), ref = "2017")
#   data$treatment <- data$status_selected
#   model <- fit_did(
#     fmla = annual_change ~ 1 + year + treatment * post,
#     pretest_fmla = annual_change ~ 1 + year * treatment,
#     data = data,
#     pretest_cols = c(
#       "year2014:treatment=0",
#       "year2015:treatment=0",
#       "year2016:treatment=0"
#     ),
#     index = c("zip_code", "year")
#   )
#   model_cov <- fit_did(
#     pretest_fmla = annual_change ~ 1 + year * treatment + treatment * post + year * (
#       log_median_household_income + total_housing +
#         pct_white + pct_higher_ed + pct_rent +
#         pct_native_hc_covered + pct_poverty + pct_supplemental_income +
#         pct_employed),
#     fmla = annual_change ~ 1 + treatment * post + year *
#       (log_median_household_income +
#         total_housing + pct_white + pct_higher_ed + pct_rent +
#         pct_native_hc_covered + pct_poverty + pct_supplemental_income + pct_employed),
#     pretest_cols = c(
#       "year2014:treatment=0",
#       "year2015:treatment=0",
#       "year2016:treatment=0"
#     ),
#     data = data,
#     index = c("zip_code", "year")
#   )
#   model_lst <- rbind(model_lst, list(nocov=model, cov=model_cov))
# }

# %%
models = [[None, None] for _ in range(4)]

models[0][0] = %R model_lst[1, "nocov"]
models[0][1] = %R model_lst[1, "cov"]
models[1][0] = %R model_lst[2, "nocov"]
models[1][1] = %R model_lst[2, "cov"]
models[2][0] = %R model_lst[3, "nocov"]
models[2][1] = %R model_lst[3, "cov"]
models[3][0] = %R model_lst[4, "nocov"]
models[3][1] = %R model_lst[4, "cov"]

# %%
result_dict = {}
for i, (m, m_cov) in enumerate(models):
    tau_q, se_q, _, pval_q = np.array(m[0].rx["coeftest_model"][0])[4, :]
    tau_cov_q, se_cov_q, _, pval_cov_q = np.array(m_cov[0].rx["coeftest_model"][0])[
        4, :
    ]

    pre_q = np.array(m[0].rx["lh_pretest"][0])[-1, -1]
    pre_q_cov = np.array(m_cov[0].rx["lh_pretest"][0])[-1, -1]

    result_dict[f"\\midrule Quartile {i+1}"] = pd.DataFrame(
        {
            "No Covariates": [
                tau_q,
                f"({se_q})",
                pval_q,
                pre_q,
                by_quartile[i]["zip_code"].nunique(),
            ],
            "Covariates": [
                tau_cov_q,
                f"({se_cov_q})",
                pval_cov_q,
                pre_q_cov,
                by_quartile[i]["zip_code"].nunique(),
            ],
        },
        index=["$\\hat \\tau$", "", "$p$-value", "Pre-trend test $p$-value", "$N$"],
    )

# %%
_ = pd.concat(result_dict).to_latex_table(
    caption="By percent employment population quartile",
    label=f"by_percent_employment_population_quartile{file_suffix}",
    filename=f"exhibits/by_pct_employment{file_suffix}.tex",
    to_latex_args=dict(column_format="llcc"),
)

# %%
for q in by_quartile:
    print(
        (
            q[["zip_code", "status_selected"]].drop_duplicates()["status_selected"]
            > 0.0
        ).mean()
    )

# %%
q1_two_period = (
    q1.query("year == 2017 or year == 2018")
    .assign(
        treatment=lambda x: x.status_selected
        >= x.status_selected.quantile(1 - pct_treated)
    )
    .set_index("zip_code")
)

# %%
pre = q1_two_period.query("year == 2017").sort_values("zip_code")
post = q1_two_period.query("year == 2018").sort_values("zip_code")
covar = (
    q1_two_period[covs]
    .reset_index()
    .drop_duplicates()
    .sort_values("zip_code")
    .drop("zip_code", axis=1)
)

# %%
null = (
    pre[["annual_change"]].isnull().any(axis=1).values
    | post[["annual_change"]].isnull().any(axis=1).values
    | covar.isnull().any(axis=1).values
)
pre = pre[~null].copy()
post = post[~null].copy()
covar = covar[~null].copy()

# %%
# %Rpush pre post covar

# %% {"language": "R"}
# library(DRDID)
# pre <- as.data.frame(pre)
# post <- as.data.frame(post)
# covariates <- as.data.frame(covar)
#
# drdid_out <- drdid_imp_panel(
#     y1 = post$annual_change,
#     y0 = pre$annual_change,
#     D = post$treatment,
#     covariates = covariates,
# )
#
# drdid_out

# %%
