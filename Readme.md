## capstone-masai

# Part A — Data Foundation and Delay Evidence

**Data Findings:**

(i) which service type is delayed most often and at what rate
    The most delayed serviced seems to be "Express" with a delay rate of 0.4897.

(ii) how delay rate changes from Tier 1 to Tier 3
    Delay rate in tier1 regions is very low at 0.04.
    Delay rate in tier3 regions is 0.44, almost half of the deliveries seem to be delayed.

(iii) how it changes across distance bands
    At lowest distance band (0-200km) has a very low delay rate of 0.0219.
    AS the distance increases, delay rate keeps increasing as well.
    At 600km+ distance, delay rate is 0.8523, which means almost all deliveries are delayed.

(iv) the three columns most strongly correlated with delivery_hours — rank the columns of part_a_correlations.csv by absolute correlation, exclude delivery_hours itself, and quote each signed value to four decimals
1. promised_hours = 0.837
2. is_delayed - distance_km = 0.545
3. distance_km = 0.401

# Part B — Delivery-Hours Regression

**Model Results(Regression)**

Baseline r2 = -0.0160
Oddly enough Baseline was the only result with a negative r2 score. r2 score being close to 0 means that val predictions end up being fairly close to the mean.

From validation set, LinearRegression comes out to be the best model, although it ran close with GradientBoostRegressor.
LinearRegression had RMSE of 4.5235

Decision tree regressor showed highest bias at unlimited depth with train_rmse = 0
it also showed high variance at depth = 1 with train_rmse = 11.6334
val_rmse - train_rmse = ~1.8 for depth = 1
val_rmse - train_rmse = 7.1727 for depth = unlimited

For GradientBoostRegressor the best learning rate was 0.1
with learning_rate = 0.01, the model was too slow and had high rmse, as well as r2 score at 0.7159
with learning_rate = 1.0, the model showed rmse and r2 score slight worse than it being 0.1, and thus it was overshooting.

in cross validation score, GradientBoostRegressor performed slightly better than LinearRegression.
Which is contrary to the results found in validation scores.
Shipping either of those models would be fine since both are performing well.
I would ship LinearRegression in this case. Since both are performing similarly well, its better to ship LinearRegression model because its comparatively easier and cheaper to train, would take time and cost less in server costs.

# Part C — Delay Classifier and Alert Threshold
**Model Results(Classification)**

- Blind Baseline: Blind Baseline has no FP and TP which means it never predicted a positive outcome at all, and thus doesn't have any score for precision, recall, and f1. Thus Blind Baseline is a useless measure in this case.

The 5 noise columns have a combines importance score of 0.2245. They all fall outside the top 7 by f-score.

K-table shows that the best validation f1 score was for k=7 while the worst was k=5, which shows that k=7 is likely the best choice in this case.

In case of tuning, the best model was the untuned model with 7 features.
The validation set in this case wasn't best used for tuning, letting the model run on the entire training set helped the model make the best generalizations.

The chosen threshold was 5 with precision at 0.79 and recall at 0.83. This threshold having the best f1_score meant that this threshold will likely give the best results. With higher threshold showing high precision but lower recall and vice versa.

In the final test with test data the scores were:
- accuracy = 0.87
- precision = 0.69
- recall = 0.75
- f1 = 0.72
With slightly higher FP than validation, and other confusion matrix results being relatively similar to validation test.
With only 41 delayed parcels its difficult to really gauge how this model will perform in the wild, and precision being below 0.7 is slightly concerning.
But accuracy being at 0.87 means the model is fairly confident in its predictions.