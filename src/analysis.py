"""
CO2 emissions and economic development: an Environmental Kuznets Curve check.

Question: as countries get richer, do their CO2 emissions per person keep rising,
or do they eventually fall (an inverted U, the "Environmental Kuznets Curve")?

Pipeline:
  1. Load raw data (Our World in Data CO2 dataset + ISO country/region list)
  2. Clean: keep real countries, years 1990-2022, drop missing values
  3. Merge the two datasets on the ISO country code and check the merge
  4. Descriptive statistics by world region
  5. Figures
  6. Regressions (OLS with year fixed effects, then country + year fixed effects,
     standard errors clustered by country)

Run from the project root:  python src/analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

START_YEAR, END_YEAR = 1990, 2022


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
OWID_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
owid_file = RAW / "owid-co2-data.csv"
if not owid_file.exists():  # the raw file (14 MB) is not stored in the repo: download it once
    print("Downloading OWID CO2 data...")
    pd.read_csv(OWID_URL).to_csv(owid_file, index=False)
co2_raw = pd.read_csv(owid_file)
regions_raw = pd.read_csv(RAW / "regions.csv", keep_default_na=False)
print(f"Raw CO2 data: {co2_raw.shape[0]:,} rows, {co2_raw.shape[1]} columns")


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
cols = ["country", "iso_code", "year", "population", "gdp", "co2", "co2_per_capita"]
co2 = co2_raw[cols].copy()

# OWID mixes countries with aggregates ("World", "Europe", "High-income countries").
# Real countries have a 3-letter ISO code; aggregates have none or start with "OWID_".
is_country = co2["iso_code"].notna() & ~co2["iso_code"].str.startswith("OWID_", na=False)
co2 = co2[is_country]

co2 = co2[co2["year"].between(START_YEAR, END_YEAR)]

n_before = len(co2)
co2 = co2.dropna(subset=["population", "gdp", "co2"])
print(f"Dropped {n_before - len(co2):,} country-years with missing GDP, CO2 or population")

# Consistency checks: no duplicate country-year, no negative values
assert not co2.duplicated(["iso_code", "year"]).any(), "duplicate country-year rows"
assert (co2[["population", "gdp", "co2"]] >= 0).all().all(), "negative values found"

# Build the variables used in the analysis
co2["gdp_pc"] = co2["gdp"] / co2["population"]          # GDP per capita (2011 international $)
co2["co2_pc"] = co2["co2"] * 1e6 / co2["population"]     # tonnes of CO2 per person (co2 is in Mt)
co2 = co2[co2["co2_pc"] > 0]                             # needed for the log
co2["log_gdp_pc"] = np.log(co2["gdp_pc"])
co2["log_gdp_pc_sq"] = co2["log_gdp_pc"] ** 2
co2["log_co2_pc"] = np.log(co2["co2_pc"])

# Sanity check: our per-capita figure should match the one OWID provides
gap = (co2["co2_pc"] - co2["co2_per_capita"]).abs().max()
print(f"Max gap between computed and OWID CO2 per capita: {gap:.4f} t")


# ---------------------------------------------------------------------------
# 3. Merge with regions
# ---------------------------------------------------------------------------
regions = regions_raw.rename(columns={"alpha-3": "iso_code"})[["iso_code", "region", "sub-region"]]
# The ISO list leaves Taiwan's region blank; fill it by hand so it is not dropped.
regions.loc[regions["iso_code"] == "TWN", ["region", "sub-region"]] = ["Asia", "Eastern Asia"]
assert (regions["region"] != "").sum() > 0
df = co2.merge(regions, on="iso_code", how="left", validate="many_to_one", indicator=True)

unmatched = df.loc[df["_merge"] == "left_only", "country"].unique()
print(f"Countries without a region after merge: {len(unmatched)} {list(unmatched)}")
df = df[df["_merge"] == "both"].drop(columns="_merge")

print(f"Final panel: {df['iso_code'].nunique()} countries, "
      f"{df['year'].min()}-{df['year'].max()}, {len(df):,} observations")
df.to_csv(ROOT / "data" / "panel_clean.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Descriptive statistics
# ---------------------------------------------------------------------------
latest = df[df["year"] == END_YEAR]
desc = (
    latest.groupby("region")
    .agg(
        countries=("iso_code", "nunique"),
        gdp_pc_median=("gdp_pc", "median"),
        co2_pc_median=("co2_pc", "median"),
        co2_pc_mean=("co2_pc", "mean"),
        total_co2_mt=("co2", "sum"),
    )
    .round(1)
    .sort_values("co2_pc_median", ascending=False)
)
desc.to_csv(TAB / f"descriptives_by_region_{END_YEAR}.csv")
print(f"\nDescriptive statistics by region ({END_YEAR}):\n{desc}\n")


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
REGION_COLORS = {
    "Africa": "#C2410C", "Americas": "#2563EB", "Asia": "#16A34A",
    "Europe": "#7C3AED", "Oceania": "#0891B2",
}
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})

# 5a. Cross-section: income vs emissions in the latest year
fig, ax = plt.subplots(figsize=(8, 5.5))
for region, g in latest.groupby("region"):
    ax.scatter(g["gdp_pc"], g["co2_pc"], s=np.sqrt(g["population"]) / 60,
               alpha=0.7, label=region, color=REGION_COLORS.get(region, "grey"),
               edgecolor="white", linewidth=0.5)
for name in ["United States", "China", "India", "France", "Qatar", "Nigeria", "Germany", "Vietnam"]:
    row = latest[latest["country"] == name]
    if not row.empty:
        ax.annotate(name, (row["gdp_pc"].iloc[0], row["co2_pc"].iloc[0]),
                    fontsize=7.5, xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("GDP per capita (international $, log scale)")
ax.set_ylabel("CO2 emissions per capita (tonnes, log scale)")
ax.set_title(f"Richer countries emit more CO2 per person ({END_YEAR})", loc="left")
ax.legend(frameon=False, fontsize=8, markerscale=0.6)
fig.tight_layout()
fig.savefig(FIG / "fig1_income_vs_co2.png")
plt.close(fig)

# 5b. Trajectories over time for a few countries
fig, ax = plt.subplots(figsize=(8, 5))
for name in ["United States", "Germany", "France", "China", "India", "Vietnam"]:
    g = df[df["country"] == name].sort_values("year")
    ax.plot(g["gdp_pc"], g["co2_pc"], marker="o", markersize=2, linewidth=1.5, label=name)
    ax.annotate(str(int(g["year"].iloc[-1])), (g["gdp_pc"].iloc[-1], g["co2_pc"].iloc[-1]),
                fontsize=7, xytext=(3, 0), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("GDP per capita (international $, log scale)")
ax.set_ylabel("CO2 emissions per capita (tonnes)")
ax.set_title(f"Development paths, {START_YEAR}-{END_YEAR}: rich countries are decoupling", loc="left")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_country_paths.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Regressions: Environmental Kuznets Curve
#    log(CO2 pc) = b1 * log(GDP pc) + b2 * log(GDP pc)^2 + fixed effects + error
#    An inverted U means b1 > 0 and b2 < 0; the turning point is exp(-b1 / (2*b2)).
# ---------------------------------------------------------------------------
cluster = {"groups": pd.factorize(df["iso_code"])[0]}

m1 = smf.ols("log_co2_pc ~ log_gdp_pc + log_gdp_pc_sq + C(year)", data=df).fit(
    cov_type="cluster", cov_kwds=cluster)
m2 = smf.ols("log_co2_pc ~ log_gdp_pc + log_gdp_pc_sq + C(year) + C(iso_code)", data=df).fit(
    cov_type="cluster", cov_kwds=cluster)


def summarize(model, name, fe):
    b1, b2 = model.params["log_gdp_pc"], model.params["log_gdp_pc_sq"]
    turning = np.exp(-b1 / (2 * b2)) if b2 < 0 else np.nan
    return {
        "model": name,
        "fixed_effects": fe,
        "b_log_gdp_pc": round(b1, 3),
        "se_log_gdp_pc": round(model.bse["log_gdp_pc"], 3),
        "b_log_gdp_pc_sq": round(b2, 3),
        "se_log_gdp_pc_sq": round(model.bse["log_gdp_pc_sq"], 3),
        "p_value_sq": round(model.pvalues["log_gdp_pc_sq"], 4),
        "turning_point_gdp_pc": round(turning, 0),
        "n_obs": int(model.nobs),
        "r2": round(model.rsquared, 3),
    }


results = pd.DataFrame([
    summarize(m1, "(1) Pooled OLS", "Year"),
    summarize(m2, "(2) Two-way FE", "Country + Year"),
])
results.to_csv(TAB / "ekc_regressions.csv", index=False)
print("Environmental Kuznets Curve regressions (SE clustered by country):")
print(results.T.to_string(header=False))

# 6b. Figure: fitted curve from the two-way FE model
b1, b2 = m2.params["log_gdp_pc"], m2.params["log_gdp_pc_sq"]
grid = np.linspace(df["log_gdp_pc"].min(), df["log_gdp_pc"].max(), 200)
shape = b1 * grid + b2 * grid ** 2
shape -= shape.max()  # normalise: 0 at the peak, so the y-axis reads as % below peak
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(np.exp(grid), 100 * (np.exp(shape) - 1), color="#2563EB", linewidth=2)
tp = np.exp(-b1 / (2 * b2))
ax.axvline(tp, color="grey", linestyle="--", linewidth=1)
ax.annotate(f"turning point ≈ ${tp:,.0f}", (tp, -12), fontsize=8, xytext=(5, 0), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("GDP per capita (international $, log scale)")
ax.set_ylabel("CO2 per capita, % below peak")
ax.set_title("Estimated within-country income-emissions curve (two-way FE)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig3_ekc_fitted.png")
plt.close(fig)

print("\nDone. Figures in outputs/figures, tables in outputs/tables.")
