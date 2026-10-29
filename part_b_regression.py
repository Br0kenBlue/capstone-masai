import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import csv


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


data = pd.read_csv("./data/modelling_table.csv")


X = data[FEATURES]
y = data["delivery_hours"]

x_train, x_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_val_scaled = scaler.transform(x_val)

def checks():
    df = pd.DataFrame({
        "metric":["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
        "value":[len(x_train), len(x_val), len(x_test), np.abs(x_train_scaled.mean(axis=0)).max(), np.abs(x_test_scaled.mean(axis=0)).max()],
    })

    df.to_csv("./outputs/part_b_checks.csv", index=False)

def baseline():
    baseline_mean = y_train.mean()

    y_val_pred = np.full(len(y_val), baseline_mean)

    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    return(["mean_baseline", "val", mae, mse, rmse, r2])
    
def dtr_calc():
    depth = [1,2,3,4,6,8,12,None]

    train_rmse_list = []
    val_rmse_list = []

    for d in depth:
        model = DecisionTreeRegressor(max_depth=d, random_state=42)
        model.fit(x_train_scaled, y_train)

        y_train_pred = model.predict(x_train_scaled)
        y_val_pred = model.predict(x_val_scaled)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

    df = pd.DataFrame({
        "max_depth":[1,2,3,4,6,8,12,"unlimited"],
        "train_rmse":train_rmse_list,
        "val_rmse":val_rmse_list,
    })

    df.to_csv("./outputs/part_b_tree_depth.csv", index=False)

#dtr depth = 6

def candidate_models():
    column_names = ["model", "split", "MAE", "MSE", "RMSE", "R2"]
    results = []
    models = {
        "LinearRegression":LinearRegression(),
        "DecisionTreeRegressor":DecisionTreeRegressor(max_depth=6, random_state=42),
        "RandomForestRegressor":RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostRegressor":GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    for name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        pipeline.fit(x_train, y_train)

        y_val_pred = pipeline.predict(x_val)

        mae = mean_absolute_error(y_val, y_val_pred)
        mse = mean_squared_error(y_val, y_val_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_val_pred)

        results.append([name, "val", mae, mse, rmse, r2])

    results.append(baseline())

    df = pd.DataFrame(results, columns=column_names)

    df.to_csv("./outputs/part_b_metrics.csv", index=False)

def lr_experiment():
    lr = [0.01, 0.1, 1]
    results = []
    column_names = ["learning_rate", "val_rmse", "val_r2"]
    for i in lr:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=i, random_state=42))
        ])
        pipeline.fit(x_train, y_train)

        y_val_pred = pipeline.predict(x_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)

        results.append([i, rmse, r2])

    df = pd.DataFrame(results, columns=column_names)

    df.to_csv("./outputs/part_b_gb_learning_rate.csv", index=False)

def cv_comp():
    models = {
        "LinearRegression":LinearRegression(),
        "GradientBoostRegressor":GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,random_state=42)}

    columns_names = ["model", "mean_r2", "std_r2"]
    results = []

    x_fold = pd.concat([x_train, x_val], axis=0)
    y_fold = pd.concat([y_train, y_val], axis=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, x_fold, y_fold, cv=kf)
        results.append([name, scores.mean(), scores.std()])

    df = pd.DataFrame(results, columns=columns_names)
    df.to_csv("./outputs/part_b_cv.csv", index=False)

def final_model():
    # Final model is LinearRegression with RMSE of 4.523542229931055
    pipeline = Pipeline([
        ("scaler",StandardScaler()),
        ("model",LinearRegression()),
    ])
    pipeline.fit(x_train, y_train)

    y_test_pred = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    res = ["LinearRegression","test",mae,mse,rmse,r2]

    with open('./outputs/part_b_metrics.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(res)

checks()
dtr_calc()
candidate_models()
lr_experiment()
cv_comp()
final_model()
    