PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv("./data/modelling_table.csv")


X = data[FEATURES]
y = data["is_delayed"]

x_train, x_tmp, y_train, y_tmp = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp,test_size=0.5,random_state=42,stratify=y_tmp)

# confusion matrix for all involved:
def cm_row(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    return{"TN":int(tn), "FP":int(fp), "FN":int(fn), "TP":int(tp),
           "accuracy":accuracy_score(y_true, pred),
           "precision":precision_score(y_true, pred, zero_division=0),
           "recall":recall_score(y_true, pred, zero_division=0),
           "f1":f1_score(y_true, pred, zero_division=0)}

# Baseline and three classifiers on validation
model_rows = [{"model":"blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int))}]

# Logisitc Regression:
def logreg(x_val):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    pipeline.fit(x_train, y_train)
    return(pipeline.predict(x_val))

# Decision Tree Classifier:
def dtc(d, x_val):
    model = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    model.fit(x_train, y_train)
    return(model.predict(x_val))

# Running idea for above decision tree using multiple depths:
depth = [2, 3, 4, 5, 6, 8, 10]

best_dtc, best_f = None, -1
dep = 0
for d in depth:
    dt = dtc(d, x_val)
    f = f1_score(y_val, dt)
    if f>best_f + 1e-12:
        best_dtc, best_f, dep = dt, f, d

# Random forest default:
def rfd(x_val):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return(model.predict(x_val))

# write csv for all the above models outputs
def c_models():
    model_rows.append({"model":"LogisticRegression", **cm_row(y_val, logreg(x_val))})
    model_rows.append({"model":f"decision_tree_depth_{dep}", **cm_row(y_val, best_dtc)})
    model_rows.append({"model":"random_forest_default", **cm_row(y_val, rfd(x_val))})

    df = pd.DataFrame(model_rows).round(4)
    df.to_csv("./outputs/part_c_models.csv", index=False)

# Searching for noise signals
rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train, y_train)
sk_all = SelectKBest(f_classif, k="all").fit(x_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sk_all.scores_,
                   "rf_importance": rf_default.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
noise_share = fs.loc[fs["is_noise"], "rf_importance"].sum()
print("noise share of random forest importance: ", round(noise_share,4))

# select k best
def sel():
    sel_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("sk", SelectKBest(f_classif, k=k)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe.fit(x_train, y_train)
        sel_rows.append({"k":k, "val_f1":f1_score(y_val, pipe.predict(x_val))})
    pd.DataFrame(sel_rows).round(4).to_csv("./outputs/part_c_selectk.csv", index=False)

# Tuning and looking for the best model

sk7 = SelectKBest(f_classif, k=7).fit(x_train, y_train)
xk_train, xk_val = sk7.transform(x_train), sk7.transform(x_val)

grid = {
    "n_estimators":[50, 100, 150],
    "max_depth":[4, 6, 8],
    "min_samples_leaf":[1, 5, 10],
}

gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(xk_train, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, cv=5, n_iter=9, random_state=42).fit(xk_train, y_train)
def7 = RandomForestClassifier(random_state=42).fit(xk_train, y_train)

DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

candidates = {
    "grid_search":(gs.best_params_, gs.best_score_, len(gs.cv_results_["params"]), 5, f1_score(y_val, gs.best_estimator_.predict(xk_val))),
    "randomized_search":(rs.best_params_, rs.best_score_, len(rs.cv_results_["params"]), 5, f1_score(y_val, rs.best_estimator_.predict(xk_val))),
    "default_forest_same_7_features":(DEFAULT_PARAMS, np.nan, 1, 0, f1_score(y_val, def7.predict(xk_val))),
}

tune_rows = []
for name, (params, cv_acc, n_cand, folds, vf1) in candidates.items():
    tune_rows.append({"search":name, "candidates":n_cand, "cv_folds":folds, "total_fits":(n_cand*folds) if folds else 1,
                      "best_params":json.dumps({k:(None if params[k] is None else int(params[k])) for k in sorted(params)}),
                      "best_cv_accuracy": cv_acc, "val_f1": vf1})

tuning = pd.DataFrame(tune_rows).round(4)
tuning.to_csv("./outputs/part_c_tuning.csv", index=False)

# Choose final model

priority = ["grid_search", "randomized_search", "default_forest_same_7_features"]
best_val = tuning["val_f1"].max()

chosen = next(n for n in priority if tuning.loc[tuning["search"] == n, "val_f1"].iloc[0] == best_val)
params = candidates[chosen][0]

# Final model:
K = 7
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=K)), 
    ("model", RandomForestClassifier(random_state=42, **params))
]).fit(x_train, y_train)

# alert validation on threshold
val_prob = final_model.predict_proba(x_val)[:, 1]
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

th_rows = [{"threshold":t, **cm_row(y_val, (val_prob >=t).astype(int))} for t in threshold_list]
th = pd.DataFrame(th_rows).round(4)
th.to_csv("./outputs/part_c_thresholds.csv", index=False)
ok = th.loc[th["recall"] >=0.8, "threshold"]
threshold = float(ok.max()) if len(ok) else 0.1

p_test = final_model.predict_proba(x_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **cm_row(y_test, (p_test >= threshold).astype(int))}]).round(4).to_csv("./outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("./outputs/part_c_final_config.csv", index=False)
print("chosen:", chosen, params, "threshold:", threshold)

c_models()
sel()