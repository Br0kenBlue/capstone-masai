## IIT-R-capstone-project-masai


# Data Findings:

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



# Model Results(Regression)

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


# Model Results(Classification)

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

# How to run
first execute the following code:

pip install -r requirements.txt
    This will make sure that your environment has the exact libraries installed as used in this project.

python generate_data.py
    This will generate the data as is asked.

python part_a_clean_eda.py
    This is start the first part of the project which is EDA - elementary data analysis
    it makes use of a custom file shipment_prep.py in ./src/ to clean the data as is provided.
    This will create 3 new files in ./outputs/:
        part_a_checks.csv
        part_a_correlations.csv
        part_a_delay_by_group.csv

python part_b_regression.py
    This piece of code does all the regression tasks of the project.
    It train 4 different regression models on the given data and outputs the results.
    It also runs an experimental LinearRegression model to check which learning rate works the best in this case.
    It runs a cross validation score check on the data using LinearRegression and GradientBoostRegressor with train + validation set data.
    Then finally it runs the final chosen model,
    In this case the final chosen model was LinearRegression because of best score in RMSE of 4.5235
    This file contains a lot of functions but all functions have been initialized at the bottom of the file.
    This will create following files in ./outputs/:
        part_b_checks.csv
        part_b_cv.csv
        part_b_gb_learning_rate.csv
        part_b_metrics.csv
        part_b_tree_depth.csv

python part_c_classification.py
    This piece of code does all the classification tasks of the project.
    It trains DecisionTreeClassifier, LogisticRegression, and RandomForestClassifier on the given data.
    It trains multiple DecisionTreeClassifier with different depth values and selects the best of them.
    It trains a RandomForestClassifier with selectkbest for multiple different values of k.
    Now K = 7 is chosen for the rest of the file.
    Now the training and validation data are transformed using selectkbest
    GridSeachcv and RandomizedSearchcv are also trained with a given grid. 
    A default RandomForestClassifier is also trained.
    Now we select the best params based on the previous three models GridSearchcv, RandomizedSearchcv, and RandomForestClassifier
    Now we run a final model based on the tuning results.
    Final model chosen is a RandomForestClassifier with selector using selectkbest at k=7
    This will create following files in ./outputs/:
        part_c_feature_scores.csv
        part_c_final_config.csv
        part_c_final_test.csv
        part_c_models.csv
        part_c_selectk.csv
        part_c_threshold.csv
        part_c_tuning.csv

python part_d_segments.py
    This file performs a segment check for the given data.
    Segments check means to verify whether the segments provided in hubs.csv are factually correct.
    This will create following files in ./outputs/:
        hub_profile.csv
        part_d_separation.csv
        segment_profile.csv
        tier_vs_segment_mismatch.csv

python package_model.py
    This will train a RandomForestClassifier with selecKbest selector in a pipeline for the given data.
    Then this will create a model dump using joblib.
    This will create following files in ./models/:
        delay_model.pkl
        delay_threshold.csv
        feature_order.csv
        impute_values.csv
    This will create following files in ./outputs/:
        part_d_reload_check.csv

python predict_delay.py
    This is the final part of the project that will take in ./data/new_shipments.csv as the test data
    This will load the model that we created earlier and use that model to predict delaty in new_shipments data
    This will only create one new fille that is:
        ./outputs/new_shipment_actions.csv

**Final running instructions in order:**

pip install -r requirements.txt
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py

# Operations recommendation
Escalate segments   -   delay rate:
Hub13   -   0.46
Hub15   -   0.36
Hub17   -   0.46
Hub18   -   0.38
Hub19   -   0.62
Hub20   -   0.48
Hub21   -   0.40
Hub23   -   0.44
Hub24   -   0.54

Tier mismatches:
HUB07,Watch,Kolkata,1
HUB13,Escalate,Nagpur,2
HUB15,Escalate,Kochi,2
HUB22,Watch,Dehradun,3

These tiers should keep track of their deliveries more closely, hire more drivers or get more delivery vehicles if needs. Or change their tier.

Delayed deliveries:
For every 10 flagged delayed deliveries, 6.9 were actually delayed.
For every 10 delayed parcels, 7.5 are caught.

Of the 48 new parcels, 8 were flagged to be delayed, with all of them needing to be escalated to the hub manager.

Why K=3?:
    Even though K=4 and K=5 had shown slightly better results, both of them had at least one cluster that was smaller than 6.
    For a dataset of 24, it having a cluster with cluster size smaller than 6 means that those clusters were not meaningfully distinct enough. And thus, K=3 was considered a better choice.

# Production plan
**Production readiness**
1. Accuracy - In part_c_final_test.csv the accuracy score is 0.87, hence the project can be said to be fairly accurate.
2. Reproducibility - All the functions work without hard coding and any results from this project can be reproduced.
3. Maintainability - All the code in this repo can be maintained with minimal impact to the actual results. sufficient comments have been added and the documentation here has been extensive.
4. Consistency - Same steps have been taken for preprocessing during training as well as predictions. shipment_prep.py handles the data cleaning in the same manner regardless of where the data comes from.
5. Monitorabilit - All the results have been recorded in a consistent manner in csv files in an organised folder.

**Quantities to monitor**
1. Delays per month
2. Orders per week
3. Customer support calls per week

**Retraining schedule**
Models should be retrained with fresh data every 2 weeks(14 days)
During sale events such as 15-28 september; regional managers should keep a close look into any delays and keep a track of it. In case of underperformance, the models should be trained on new data every 2 days.
Compare the data with similar sale events from the past.
*Data-drift*: Data drift here can be due to mismatch in data representations and the features used in different regions.

# Limitations
*Synthetic data*: The data used in this project is synthetic, as such this is not an accurate representation of the real world application.
The number of delayed parcels is also quite low(10) so it does not provide the best representation of real world application.

*Real-world limitations*: Training data here cannot fully capture the real world limitations for a delivery company.Many factors such as traffic conditions, time of the day for delivery run with their respective locations, driver experience, Number of drivers/vehicles and their availability, time of the year, weather related delays etc. were not accounted for in this project. Those factors can often time be make or break for delivery companies.
A better way to work would be by comparing the data for deliveries at different time of the year and understand the pattern, thus making different predictions and different recommendations based on dynamic conditions.

*Very few actual delays*: Very few acual delays are present in the test data set i.e. new_shipments.csv. As such the systems in this project did not get tested to their limits. It's hard to say that this system will be reliable when its deployed in production because the testing was not extensive enough with only 10 delays being present.

*Predictive reasoning*: Some places can have more delays due to rain whereas in some places rain doesn't affect the deliveries at all, but when a regression model is training it only sees the rain data and not that it affects one place more than the others. This sort of data mismatch is difficult to handle and thus this repo does not address that either.