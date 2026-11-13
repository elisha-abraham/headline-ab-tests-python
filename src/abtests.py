"""Tools for analysing many A/B tests at once.

In the Upworthy archive each test compares several "packages" of the same
story: a headline with its image and teaser text. Visitors were shown one
package at random. A row is one package, with the number of times it was
shown (impressions) and clicked.

Everything here is plain numpy, pandas and scipy, so each step can be read.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import optimize, stats

# What can differ between the packages of a test, grouped into three kinds.
FIELDS = {"headline": "headline", "eyecatcher_id": "image", "square": "image",
          "excerpt": "teaser text", "lede": "teaser text", "share_text": "teaser text"}


# --------------------------------------------------------------------------- data

def load(path) -> pd.DataFrame:
    """One row per package, for tests with at least two packages."""
    columns = ["clickability_test_id", "created_at", "headline", "excerpt", "lede", "eyecatcher_id", "share_text",
               "square", "impressions", "clicks", "first_place", "winner"]
    data = pd.read_csv(path, usecols=columns, low_memory=False).rename(columns={"clickability_test_id": "test"})
    assert data["impressions"].ge(1).all(), "a package with no impressions"
    assert (data["clicks"] >= 0).all() and (data["clicks"] <= data["impressions"]).all(), "more clicks than impressions"
    data = data[data.groupby("test")["test"].transform("size") >= 2].reset_index(drop=True)
    data["headline"] = data["headline"].fillna("").astype(str)
    data["ctr"] = data["clicks"] / data["impressions"]
    return data


def what_varies(data: pd.DataFrame) -> pd.Series:
    """For each test, what differs between its packages."""
    distinct = data.groupby("test")[list(FIELDS)].nunique(dropna=False) > 1
    kinds = pd.DataFrame({kind: distinct[[field for field, k in FIELDS.items() if k == kind]].any(axis=1)
                          for kind in ("headline", "image", "teaser text")})
    label = np.select([kinds.sum(axis=1) > 1, kinds["headline"], kinds["image"], kinds["teaser text"]],
                      ["several things", "headline only", "image only", "teaser text only"], "nothing recorded")
    return pd.Series(label, index=kinds.index, name="varies")


# --------------------------------------------------------------------------- one row per test

def test_table(data: pd.DataFrame) -> pd.DataFrame:
    """Per test: size, click rate, a test of whether the packages differ, and a check of the traffic split.

    p_value     chi-square test that every package in the test has the same click rate
    split_p     chi-square test that every package was shown equally often (a "sample ratio" check)
    """
    test = data["test"]
    shown, clicked = data["impressions"], data["clicks"]
    total_shown = shown.groupby(test).transform("sum")
    total_clicked = clicked.groupby(test).transform("sum")
    packages = shown.groupby(test).transform("size")
    expected = shown * total_clicked / total_shown  # clicks each package would get if all rates were equal
    with np.errstate(divide="ignore", invalid="ignore"):
        cell = (clicked - expected) ** 2 * (1 / expected + 1 / (shown - expected))
    even = total_shown / packages
    grouped = data.groupby("test")
    table = pd.DataFrame({"packages": grouped.size(), "impressions": grouped["impressions"].sum(),
                          "clicks": grouped["clicks"].sum()})
    table["ctr"] = table["clicks"] / table["impressions"]
    table["smallest_expected_clicks"] = expected.groupby(test).min()
    table["chi2"] = cell.groupby(test).sum().where(table["clicks"] > 0)
    table["p_value"] = stats.chi2.sf(table["chi2"], table["packages"] - 1)
    table["split_p"] = stats.chi2.sf(((shown - even) ** 2 / even).groupby(test).sum(), table["packages"] - 1)
    table["best_ctr"], table["worst_ctr"] = grouped["ctr"].max(), grouped["ctr"].min()
    return table


def benjamini_hochberg(p_values: pd.Series, rate: float = 0.05) -> pd.Series:
    """True for the tests that count as discoveries when the false discovery rate is held at `rate`."""
    ordered = np.sort(p_values.dropna().to_numpy())
    passed = ordered <= rate * np.arange(1, len(ordered) + 1) / len(ordered)
    cutoff = ordered[passed].max() if passed.any() else -1.0
    return p_values.le(cutoff) & p_values.notna()


def share_with_no_difference(p_values: pd.Series, threshold: float = 0.5) -> float:
    """Storey's estimate of the share of tests in which the packages truly do not differ.

    Without a real difference p-values are spread evenly between 0 and 1, so nearly every p-value
    above the threshold comes from such a test, and as many again lie below it.
    """
    p = p_values.dropna()
    return min(1.0, float((p > threshold).mean() / (1 - threshold)))


def holm(p_values: pd.Series) -> pd.Series:
    """Holm-adjusted p-values: the standard correction when a handful of hypotheses are tested together."""
    ordered = p_values.sort_values()
    count = len(ordered)
    adjusted = (ordered * (count - np.arange(count))).cummax().clip(upper=1.0)
    return adjusted.reindex(p_values.index)


# --------------------------------------------------------------------------- how big are real differences?

def log_odds(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Log odds of a click for each package, and its sampling variance (0.5 added so that zero clicks is defined).
    Click rates are about 1.5%, so a difference in log odds is very nearly a relative difference in click rate."""
    clicked, missed = data["clicks"] + 0.5, data["impressions"] - data["clicks"] + 0.5
    return np.log(clicked / missed), 1 / clicked + 1 / missed


def true_spread(data: pd.DataFrame) -> dict:
    """How much packages in the same test truly differ, after sampling noise is taken out.

    The variance of the measured log odds inside a test is true variance plus noise. The noise is known
    from the counts, so the true part is what is left, pooled over tests.
    """
    value, noise = log_odds(data)
    test = data["test"]
    weight = value.groupby(test).size() - 1
    observed = (value.groupby(test).var(ddof=1) * weight).sum() / weight.sum()
    noise_only = (noise.groupby(test).mean() * weight).sum() / weight.sum()
    return {"observed_variance": observed, "noise_variance": noise_only, "true_variance": observed - noise_only,
            "true_sd": float(np.sqrt(max(observed - noise_only, 0.0)))}


def winners(data: pd.DataFrame, true_variance: float) -> pd.DataFrame:
    """For each test, the package with the highest click rate: its measured lift over the test's average
    package, and the lift to expect from it once chance is allowed for.

    A package's measured advantage is part real, part luck. With true differences of variance T and a
    measurement of noise variance V, the best estimate of the real advantage keeps the share T / (T + V) of
    what was measured. The best package in a test was partly selected for its luck, which is why its
    measured lift overstates what it will deliver.
    """
    value, noise = log_odds(data)
    deviation = value - value.groupby(data["test"]).transform("mean")
    kept = true_variance / (true_variance + noise)
    top = deviation.groupby(data["test"]).idxmax()
    return pd.DataFrame({"test": data.loc[top, "test"].to_numpy(),
                         "measured_lift": np.exp(deviation[top].to_numpy()) - 1,
                         "expected_true_lift": np.exp((kept * deviation)[top].to_numpy()) - 1,
                         "share_kept": kept[top].to_numpy(),
                         "clicks": data.loc[top, "clicks"].to_numpy()}).set_index("test")


def lift_from_noise_alone(data: pd.DataFrame, rounds: int = 20, seed: int = 20261112) -> float:
    """Median measured lift of the "best" package if no package were truly better than another.

    Every package keeps its real number of impressions and is given its test's overall click rate; clicks
    are then drawn at random. Whatever lift the best package shows is luck.
    """
    rng = np.random.default_rng(seed)
    rate = (data["clicks"].groupby(data["test"]).transform("sum") / data["impressions"].groupby(data["test"]).transform("sum"))
    medians = []
    for _ in range(rounds):
        fake = data[["test", "impressions"]].assign(clicks=rng.binomial(data["impressions"].to_numpy(), rate.to_numpy()))
        value, _ = log_odds(fake)
        deviation = value - value.groupby(fake["test"]).transform("mean")
        medians.append(float(np.median(np.exp(deviation.groupby(fake["test"]).max()) - 1)))
    return float(np.mean(medians))


# --------------------------------------------------------------------------- power

def _z(power: float, alpha: float) -> float:
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def impressions_needed(rate: float, lift: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Impressions per package for a two-package test to detect a relative `lift` over click rate `rate`."""
    better = rate * (1 + lift)
    return _z(power, alpha) ** 2 * (rate * (1 - rate) + better * (1 - better)) / (better - rate) ** 2


def minimum_detectable_lift(impressions: float, rate: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest relative lift a two-package test with this much traffic would detect with the given power."""
    return optimize.brentq(lambda lift: impressions_needed(rate, lift, power, alpha) - impressions, 1e-4, 50)


# --------------------------------------------------------------------------- everything in one row

def summary(data: pd.DataFrame) -> pd.Series:
    """The headline numbers of one file in one place, so that two files can be compared like for like."""
    tests = test_table(data).join(what_varies(data))
    enough = tests["smallest_expected_clicks"] >= 5
    p = tests.loc[enough, "p_value"]
    spread = true_spread(data)
    best = winners(data, spread["true_variance"])
    shown, rate = float(data["impressions"].median()), float(tests["ctr"].median())
    by_kind = tests[enough].groupby("varies")["p_value"].apply(lambda values: 100 * (values < 0.05).mean())
    return pd.Series({
        "Tests": len(tests),
        "Packages": len(data),
        "Median impressions per package": shown,
        "Median click rate of a test, %": 100 * rate,
        "Tests with an uneven traffic split, %": 100 * (tests["split_p"] < 0.001).mean(),
        "Tests significant at 0.05, %": 100 * (p < 0.05).mean(),
        "Tests significant with the false discovery rate held at 5%, %": 100 * benjamini_hochberg(p).mean(),
        "Tests estimated to hold a real difference, %": 100 * (1 - share_with_no_difference(p)),
        "Headline-only tests significant at 0.05, %": by_kind.get("headline only", np.nan),
        "Image-only tests significant at 0.05, %": by_kind.get("image only", np.nan),
        "Teaser-text-only tests significant at 0.05, %": by_kind.get("teaser text only", np.nan),
        "True spread of click rates within a test (SD of log odds)": spread["true_sd"],
        "Measured lift of the best package over the test average, median %": 100 * best["measured_lift"].median(),
        "Lift of the best package from luck alone, median %": 100 * lift_from_noise_alone(data),
        "Expected true lift of the best package, median %": 100 * best["expected_true_lift"].median(),
        "Minimum detectable lift at the median traffic, %": 100 * minimum_detectable_lift(shown, rate),
    })


# --------------------------------------------------------------------------- what the headlines say

FEATURES = {
    "words_per_5": "Five more words",
    "question": "Has a question mark",
    "number": "Contains a number",
    "first_person_singular": "Says I, my or me",
    "first_person_plural": "Says we, our or us",
    "points_forward": "Points forward (this, these, here's)",
    "exclamation": "Has an exclamation mark",
    "colon": "Has a colon",
    "second_person": "Speaks to the reader (you, your)",
    "quotation": "Has a quotation",
    "starts_with_number": "Starts with a number",
    "emphasis": "Shouts (a word in capitals or *stars*)",
}


def headline_features(headlines: pd.Series) -> pd.DataFrame:
    """Simple, countable properties of a headline. Each is 0 or 1, except the word count."""
    text = headlines.fillna("").astype(str)

    def has(pattern: str, flags: int = 0) -> pd.Series:
        return text.str.contains(pattern, flags=flags, regex=True).astype(float)

    return pd.DataFrame({
        "words_per_5": text.str.split().str.len() / 5,
        "question": has(r"\?"),
        "number": has(r"\d"),
        "first_person_singular": has(r"\b(?:I|I'm|I've|I'd|I'll|[Mm]y|[Mm]e|[Mm]yself)\b"),
        # "US" in capitals is the country, so only "us" and "Us" count as the pronoun.
        "first_person_plural": has(r"\b(?:[Ww]e|[Ww]e're|[Ww]e've|[Ww]e'll|[Oo]ur|[Uu]s)\b"),
        "points_forward": has(r"\b(?:this|these|here's|here is|here are)\b", re.IGNORECASE),
        "exclamation": has("!"),
        "colon": has(":"),
        "second_person": has(r"\b(?:you|your|yours|yourself|you're|you've|you'll|you'd)\b", re.IGNORECASE),
        "quotation": has("[\"“”]"),
        "starts_with_number": has(r"^\s*\d"),
        "emphasis": has(r"\*\w|\b[A-Z]{3,}\b"),
    }, index=headlines.index)


def within_test_model(data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """How a headline property changes the click rate, comparing packages only with rivals in the same test.

    Model: a test's clicks are divided among its packages in proportion to impressions x exp(x . beta).
    This is Poisson regression with a separate baseline for every test; the baselines cancel out, so nothing
    about the story, the day or the audience can explain the result. exp(beta) is a click-rate ratio.

    Standard errors are clustered by test (a sandwich estimator): clicks vary more from package to package
    than the model alone allows, and ignoring that would make every effect look more certain than it is.
    """
    codes = pd.factorize(data["test"])[0]
    tests = codes.max() + 1

    def per_test(values: np.ndarray) -> np.ndarray:
        return np.column_stack([np.bincount(codes, weights=values[:, k], minlength=tests) for k in range(values.shape[1])])

    X = features.to_numpy(dtype=float)
    X = X - (per_test(X) / np.bincount(codes, minlength=tests)[:, None])[codes]  # centring within a test changes nothing
    clicks = data["clicks"].to_numpy(dtype=float)
    offset = np.log(data["impressions"].to_numpy(dtype=float))
    test_clicks = np.bincount(codes, weights=clicks, minlength=tests)
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        score = np.exp(offset + X @ beta)
        share = score / np.bincount(codes, weights=score, minlength=tests)[codes]
        fitted = test_clicks[codes] * share
        mean_x = per_test(share[:, None] * X)
        information = (X * fitted[:, None]).T @ X - (mean_x * test_clicks[:, None]).T @ mean_x
        step = np.linalg.solve(information, X.T @ (clicks - fitted))
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    bread = np.linalg.inv(information)
    test_scores = per_test(X * (clicks - fitted)[:, None])
    se = np.sqrt(np.diag(bread @ (test_scores.T @ test_scores) @ bread))
    z = beta / se
    return pd.DataFrame({"ratio": np.exp(beta), "low": np.exp(beta - 1.96 * se), "high": np.exp(beta + 1.96 * se),
                         "beta": beta, "se": se, "z": z, "p_value": 2 * stats.norm.sf(np.abs(z)),
                         "model_se": np.sqrt(np.diag(bread))}, index=features.columns)
