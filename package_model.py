import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, clean_shipments

data = pd.read_csv("./outputs/part_c_final_config.csv").iloc[0]
max_depth = None if data["max_depth"] == "unlimited" else int(data["max_depth"])

table = pd.read_csv("./data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]

x_train, x_tmp, y_train, y_tmp = train_test_split(X, y, train_size=0.3, random_state=42, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, train_size=0.5, random_state=42, stratify=y_tmp)

pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(data["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42, n_estimators=int(data["n_estimators"]), max_depth=max_depth, min_samples_leaf=int(data["min_samples_leaf"])))
])
pipeline.fit(x_train, y_train)

joblib.dump(pipeline, "./models/delay_model.pkl")
pd.DataFrame({"threshold":[float(data["threshold"])]}).to_csv("./models/delay_threshold.csv", index=False)
pd.DataFrame({"feature":FEATURES}).to_csv("models/feature_order.csv", index=False)
i, medians = clean_shipments(pd.read_csv("./data/raw_shipments.csv"))
pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}).to_csv("./models/impute_values.csv", index=False)

reloaded = joblib.load("./models/delay_model.pkl")
diff = np.abs(reloaded.predict_proba(x_test)[:, 1] - pipeline.predict_proba(x_test)[:, 1]).max()
pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
              "value": [len(x_test), diff, float(data["threshold"])]}).round(4).to_csv("outputs/part_d_reload_check.csv", index=False)