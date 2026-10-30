PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

def cm_row(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    return{"TN":int(tn), "FP":int(fp), "FN":int(fn), "TP":int(tp),
           "accuracy":accuracy_score(y_true, pred),
           "precision":precision_score(y_true, pred, zero_division=0),
           "recall":recall_score(y_true, pred, zero_division=0),
           "f1":f1_score(y_true, pred, zero_division=0)}

# Baseline and three classifiers on validation
model_rows = [{"model":"blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int))}]

def logreg(x_val):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    pipeline.fit(x_train, y_train)
    return(pipeline.predict(x_val))

def dtc(d, x_val):
    model = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    model.fit(x_train, y_train)
    return(model.predict(x_val))

depth = [2, 3, 4, 5, 6, 8, 10]

best_dtc, best_f = None, -1
dep = 0
for d in depth:
    dt = dtc(d, x_val)
    f = f1_score(y_val, dt)
    if f>best_f + 1e-12:
        best_dtc, best_f, dep = dt, f, d

def rfd(x_val):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return(model.predict(x_val))

def c_models():
    model_rows.append({"model":"LogisticRegression", **cm_row(y_val, logreg(x_val))})
    model_rows.append({"model":f"decision_tree_depth_{dep}", **cm_row(y_val, best_dtc)})
    model_rows.append({"model":"random_forest_default", **cm_row(y_val, rfd(x_val))})

    df = pd.DataFrame(model_rows).round(4)
    df.to_csv("./outputs/part_c_models.csv", index=False)

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train, y_train)
sk_all = SelectKBest(f_classif, k="all").fit(x_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sk_all.scores_,
                   "rf_importance": rf_default.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
noise_share = fs.loc[fs["is_noise"], "rf_importance"].sum()


sel_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("sk", SelectKBest(f_classif, k=k)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(x_train, y_train)
    sel_rows.append({"k":k, "val_f1":f1_score(y_val, pipe.predict(x_val))})
pd.DataFrame(sel_rows).round(4).to_csv("./outputs/part_c_selectk.csv", index=False)


#c_models()