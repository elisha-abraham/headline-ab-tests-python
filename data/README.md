# Data notes

## Source

Matias, J. N., Munger, K., Aubin Le Quere, M. S. A., & Ebersole, C. R. *The Upworthy Research Archive*. OSF. https://osf.io/jd64p

The archive holds the headline and image experiments run by the news site Upworthy between January 2013 and April 2015. For each story the editors wrote several "packages" (a headline with an image and teaser text); visitors were shown one package at random, and the site counted impressions and clicks.

The archive is published in parts so that analysts can explore one part and test what they find on another. This project uses two of them. The files are not stored in this repository (81 MB); the download script fetches them from OSF.

```bash
python scripts/01_download_data.py
```

| File | Packages | Tests | Used for |
|---|---|---|---|
| `upworthy-archive-exploratory-packages-03.12.2020.csv` (14.3 MB) | 22,666 | 4,873 | Exploration: anything may be tried |
| `upworthy-archive-confirmatory-packages-03.12.2020.csv` (66.5 MB) | 105,551 | 22,743 | One pre-planned test of what exploration found |

`data/raw/manifest.json` records the size and SHA-256 of each file as downloaded.

## Columns used

| Column | Meaning |
|---|---|
| `clickability_test_id` | Identifies the test. Packages with the same value competed against each other |
| `headline`, `excerpt`, `lede`, `share_text` | The text of the package: headline and teaser texts |
| `eyecatcher_id`, `square` | Identify the images of the package |
| `impressions` | How many visitors were shown the package |
| `clicks` | How many of them clicked |
| `created_at` | When the package was created |

A test is classified by what differs between its packages: the headline only, the image only, the teaser text only, several of these, or nothing recorded.

## Result files in `results/`

| File | Holds |
|---|---|
| `exploratory_summary.csv`, `confirmatory_summary.csv` | The headline figures of each file |
| `predictions.csv` | The exploratory figures and how far the confirmatory ones may differ and still replicate |
| `replication.csv` | Each prediction against the confirmatory figure |
| `exploratory_headline_model.csv` | The headline model on the exploratory tests, and which properties became hypotheses |
| `confirmatory_headline_model.csv` | The same model on the confirmatory tests, with the verdict on each hypothesis |
| `effect_shrinkage.csv` | Size of each hypothesised effect before and after |
| `exploratory_tests_by_kind.csv`, `sample_sizes.csv` | Share of tests significant by what was varied; traffic needed to detect a given lift |
