# Analysis plan for the confirmatory file

Written after exploring the exploratory file (4,873 tests) and before computing anything from the confirmatory file (22,743 tests). The commit history of this repository shows the order: this plan was committed before `02_confirmatory.py` was run.

What I had seen of the confirmatory file when this was written: its size, its column names and its first six rows, when checking the download. Nothing had been computed from its clicks.

## Why a plan

The exploratory notebook tried twelve properties of a headline and kept the seven that looked strongest. Picking the strongest results out of many guarantees that some of them are flattered by chance. The only honest test is new data and a rule fixed in advance.

## A. Predictions about the tests as a whole

If the exploratory file is a fair sample of Upworthy's tests, the confirmatory file should give nearly the same figures. Each prediction replicates if the confirmatory figure lands within the allowed difference. The figures and allowances are stored in `results/predictions.csv`.

| Measure | Exploratory | Allowed difference |
|---|---|---|
| Tests with an uneven traffic split | 15.5% | 3 points |
| Tests significant at 0.05 | 45.3% | 3 points |
| Tests significant with the false discovery rate held at 5% | 35.3% | 3 points |
| Tests estimated to hold a real difference | 64.4% | 4 points |
| Headline-only tests significant at 0.05 | 55.6% | 4 points |
| Image-only tests significant at 0.05 | 43.0% | 4 points |
| Teaser-text-only tests significant at 0.05 | 23.3% | 4 points |
| True spread of click rates within a test (SD of log odds) | 0.244 | 0.02 |
| Measured lift of the best package, median | 29.9% | 3 points |
| Lift of the best package from luck alone, median | 16.5% | 2 points |
| Expected true lift of the best package, median | 21.2% | 3 points |
| Minimum detectable lift at the median traffic | 71.6% | 5 points |

## B. Hypotheses about headlines

Sample: every test in the confirmatory file in which the headline is the only thing that differs between packages.

Model: the one in `src/abtests.py` (`within_test_model`), with all twelve properties entered together, exactly as defined in `headline_features` at the commit of this plan. Standard errors clustered by test.

| # | Property of the headline | Predicted effect on clicks | Exploratory ratio |
|---|---|---|---|
| H1 | Has a question mark | fewer | 0.902 |
| H2 | Five more words | more | 1.068 |
| H3 | Says we, our or us | fewer | 0.919 |
| H4 | Has a colon | fewer | 0.927 |
| H5 | Contains a number | more | 1.063 |
| H6 | Has an exclamation mark | fewer | 0.895 |
| H7 | Points forward (this, these, here's) | more | 1.045 |

Decision rule: a hypothesis is confirmed if its two-sided p-value, adjusted by Holm's method across the seven, is below 0.05 **and** the effect is in the predicted direction.

The other five properties (says I, my or me; speaks to the reader; has a quotation; starts with a number; shouts) carry no prediction. They are estimated and reported for completeness.

## C. One more prediction

The seven hypotheses were chosen because their exploratory effects were large. For the same reason that a winning package's lift is partly luck, their confirmatory effects should on average be closer to 1 than the exploratory ones.

## D. What will not be done

- No property will be added, dropped or redefined after the confirmatory results are seen.
- No subgroup will be analysed to rescue a hypothesis that fails.
- Anything further that is tried on the confirmatory file will be labelled as exploratory.
