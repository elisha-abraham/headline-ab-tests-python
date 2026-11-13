# %% [markdown]
# # What thousands of headline experiments say about A/B testing
# ## Part 2 of 2: confirmation
#
# The exploratory notebook made predictions from 4,873 tests. This notebook tests them on the 22,743 tests
# of the confirmatory file, following `analysis_plan.md`, which was committed before this notebook was run.
# Nothing here was tuned after seeing these results.
#
# This notebook is generated from `02_confirmatory.py` by `scripts/build_notebooks.py`.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "abtests.py").exists())
sys.path.insert(0, str(ROOT / "src"))
import abtests as ab  # noqa: E402
import charts  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
pd.set_option("display.width", 170, "display.max_columns", 20, "display.max_colwidth", 95,
              "display.float_format", "{:,.3f}".format)

data = ab.load(ROOT / "data" / "raw" / "upworthy-archive-confirmatory-packages-03.12.2020.csv")
tests = ab.test_table(data).join(ab.what_varies(data))
print(f"{len(data):,} packages in {len(tests):,} tests")

# %% [markdown]
# ## A. Do the figures for the tests as a whole replicate?
#
# Each prediction replicates if the confirmatory figure is within the difference allowed in the plan.

# %%
predictions = pd.read_csv(RESULTS / "predictions.csv", index_col="measure")
confirmatory = ab.summary(data)
replication = predictions.assign(confirmatory=confirmatory.reindex(predictions.index))
replication["difference"] = replication["confirmatory"] - replication["exploratory"]
replication["replicated"] = replication["difference"].abs() <= replication["allowed_difference"]
replication.to_csv(RESULTS / "replication.csv")
confirmatory.rename("confirmatory").to_csv(RESULTS / "confirmatory_summary.csv", header=True, index_label="measure")
display(replication)
print(f"{int(replication['replicated'].sum())} of {len(replication)} predictions replicated")

# %% [markdown]
# ## B. The seven hypotheses about headlines

# %%
explored = pd.read_csv(RESULTS / "exploratory_headline_model.csv", index_col="feature")
headline_tests = tests.index[tests["varies"] == "headline only"]
pure = data[data["test"].isin(headline_tests)].reset_index(drop=True)
fit = ab.within_test_model(pure, ab.headline_features(pure["headline"]))
print(f"{len(headline_tests):,} tests that vary only the headline, {len(pure):,} headlines")

result = pd.DataFrame({"property": explored["property"], "hypothesis": explored["hypothesis"],
                       "predicted": np.where(explored["beta"] > 0, "more clicks", "fewer clicks"),
                       "exploratory_ratio": explored["ratio"], "exploratory_low": explored["low"],
                       "exploratory_high": explored["high"], "confirmatory_ratio": fit["ratio"],
                       "confirmatory_low": fit["low"], "confirmatory_high": fit["high"], "p_value": fit["p_value"]})
hypotheses = result[result["hypothesis"]].copy()
hypotheses["holm_p"] = ab.holm(hypotheses["p_value"])
hypotheses["same_direction"] = (hypotheses["confirmatory_ratio"] > 1) == (hypotheses["exploratory_ratio"] > 1)
hypotheses["confirmed"] = hypotheses["same_direction"] & (hypotheses["holm_p"] < 0.05)
result = result.join(hypotheses[["holm_p", "same_direction", "confirmed"]])
result.loc[~result["hypothesis"], "predicted"] = "no prediction"
result.to_csv(RESULTS / "confirmatory_headline_model.csv", index_label="feature")
confirmed = int(hypotheses["confirmed"].sum())
display(hypotheses[["property", "predicted", "exploratory_ratio", "confirmatory_ratio", "confirmatory_low",
                    "confirmatory_high", "holm_p", "confirmed"]].sort_values("holm_p"))
print(f"{confirmed} of {len(hypotheses)} hypotheses confirmed")

# %% [markdown]
# The five properties that carried no prediction:

# %%
display(result.loc[~result["hypothesis"], ["property", "exploratory_ratio", "confirmatory_ratio", "confirmatory_low",
                                           "confirmatory_high", "p_value"]])

# %%
def row(feature: str) -> dict:
    values = result.loc[feature]
    return {"label": values["property"],
            "first": (values["exploratory_ratio"], values["exploratory_low"], values["exploratory_high"]),
            "second": (values["confirmatory_ratio"], values["confirmatory_low"], values["confirmatory_high"])}


strongest_first = explored["z"].abs().sort_values(ascending=False).index
rows = ([{"label": "Predicted to change the click rate"}] + [row(f) for f in strongest_first if result.loc[f, "hypothesis"]]
        + [{"label": "No prediction made"}] + [row(f) for f in strongest_first if not result.loc[f, "hypothesis"]])
charts.paired_forest(rows, (f"Exploratory: {int(explored['tests'].iloc[0]):,} tests", f"Confirmatory: {len(headline_tests):,} tests"),
                     f"{confirmed} of {len(hypotheses)} headline patterns found in exploration held up on new tests",
                     "Click-rate ratio for a headline with the property against a rival headline for the same story without it,\n"
                     "with 95% intervals. The number printed is the confirmatory estimate.",
                     FIGURES, "04_headline_patterns", [0.8, 0.9, 1.0, 1.1, 1.2], "Click-rate ratio (log scale)")
display(Image(filename=str(FIGURES / "04_headline_patterns_light.png"), width=720))

# %% [markdown]
# ## C. Did the effects shrink, as predicted?
#
# The plan predicted that effects chosen for being large in exploration would be closer to 1 on new data.

# %%
size_before = np.log(hypotheses["exploratory_ratio"]).abs()
size_after = np.log(hypotheses["confirmatory_ratio"]).abs() * np.where(hypotheses["same_direction"], 1, -1)
print(f"average size of the seven effects in exploration:  {100 * (np.exp(size_before.mean()) - 1):.1f}%")
print(f"average size of the same effects in confirmation: {100 * (np.exp(size_after.mean()) - 1):.1f}%")
print(f"effects smaller in confirmation: {int((size_after < size_before).sum())} of {len(hypotheses)}")
pd.DataFrame({"property": hypotheses["property"], "exploratory_size_%": 100 * (np.exp(size_before) - 1),
              "confirmatory_size_%": 100 * (np.exp(size_after) - 1)}).to_csv(RESULTS / "effect_shrinkage.csv", index_label="feature")
