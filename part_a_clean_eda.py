import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features


hubs = pd.read_csv("./data/hubs.csv")
raw = pd.read_csv("./data/raw_shipments.csv")

clean, medians = clean_shipments(raw)
final = build_features(clean, hubs)

final["is_delayed"] = (final["delivery_hours"]>final["promised_hours"]).astype(int)

def main():
    final.to_csv('./data/modelling_table.csv', index=False)

def checks():
    data = pd.read_csv("./data/modelling_table.csv")

    raw_rows = len(raw)
    duplicate_rows_removed = raw.duplicated().sum()
    rows_after_dedup = len(clean)
    missing_distance_km_before = raw["distance_km"].isnull().sum()
    missing_parcel_weight_before = raw["parcel_weight_kg"].isnull().sum()
    sentinel_weight_rows = len(raw[raw["parcel_weight_kg"]>100])
    missing_declared_value_before = raw["declared_value"].isnull().sum()
    missing_payment_mode_before = raw["payment_mode"].isnull().sum()
    service_type_spellings_before = raw["service_type"].nunique()
    service_type_spellings_after = data["service_type"].nunique()
    declared_value_mean_inr = data["declared_value_inr"].mean()
    log_declared_value_mean = data["log_declared_value"].mean()
    median_distance_km = medians["distance_km"]
    median_parcel_weight_kg = medians["parcel_weight_kg"]
    median_declared_value_inr = medians["declared_value_inr"]
    rows_after_hub_merge = len(data)
    nulls_in_features_and_target = data.isnull().sum().sum()
    delayed_count = len(data[data["is_delayed"] == 1])
    delay_rate = round(delayed_count/len(data), 4)
    hours_q1 = data["delivery_hours"].quantile(0.25)
    hours_q3 = data["delivery_hours"].quantile(0.75)
    hours_lower_bound = hours_q1 - 1.5*(hours_q3 - hours_q1)
    hours_upper_bound = hours_q3 + 1.5*(hours_q3 - hours_q1)
    hours_outlier_count = len(data[data["delivery_hours"]>hours_upper_bound])
    declared_value_median_inr = medians["declared_value_inr"]
    log_declared_value_median = data["log_declared_value"].median()

    part_a_checks = {
        "raw_rows":raw_rows,
        "duplicate_rows_removed":duplicate_rows_removed,
        "rows_after_dedup":rows_after_dedup,
        "missing_distance_km_before":missing_distance_km_before,
        "missing_parcel_weight_before":missing_parcel_weight_before,
        "sentinel_weight_rows":sentinel_weight_rows,
        "missing_declared_value_before":missing_declared_value_before,
        "missing_payment_mode_before":missing_payment_mode_before,
        "service_type_spellings_before":service_type_spellings_before,
        "service_type_spellings_after":service_type_spellings_after,
        "declared_value_mean_inr":round(declared_value_mean_inr,2),
        "log_declared_value_mean":round(log_declared_value_mean,3),
        "median_distance_km":round(median_distance_km,2),
        "median_parcel_weight_kg":median_parcel_weight_kg,
        "median_declared_value_inr":median_declared_value_inr,
        "rows_after_hub_merge":rows_after_hub_merge,
        "nulls_in_features_and_target":nulls_in_features_and_target,
        "delayed_count":delayed_count,
        "delay_rate":delay_rate,
        "hours_q1":round(hours_q1,2),
        "hours_q3":round(hours_q3,2),
        "hours_lower_bound":round(hours_lower_bound,2),
        "hours_upper_bound":round(hours_upper_bound,2),
        "hours_outlier_count":hours_outlier_count,
        "declared_value_median_inr":declared_value_median_inr,
        "log_declared_value_median":round(log_declared_value_median,3),    
    }

    df = pd.DataFrame(part_a_checks.items(), columns=['metric','value'])
    df.to_csv('./outputs/part_a_checks.csv', index=False)

def delay_evidence():
    # columns = dimension, group, shipments, delayed, delay_rate, mean_hours
    data = pd.read_csv("./data/modelling_table.csv")

    econ_total = (data["service_type"]=="Economy").sum()
    exp_total = (data["service_type"]=="Express").sum()
    std_total = (data["service_type"]=="Standard").sum()

    hub_tier1 = (data["hub_tier"] == 1).sum()
    hub_tier2 = (data["hub_tier"] == 2).sum()
    hub_tier3 = (data["hub_tier"] == 3).sum()

    dist_band1 = (data["distance_km"] <= 200).sum()
    dist_band2 = ((data["distance_km"] > 200) & (data["distance_km"] <= 400)).sum()
    dist_band3 = ((data["distance_km"] > 400) & (data["distance_km"] <= 600)).sum()
    dist_band4 = (data["distance_km"] > 600).sum()

    econ_delay = (data[data["service_type"]=="Economy"]["is_delayed"] == 1).sum()
    exp_delay = (data[data["service_type"]=="Express"]["is_delayed"] == 1).sum()
    std_delay = (data[data["service_type"]=="Standard"]["is_delayed"] == 1).sum()

    tier1_delay = (data[data["hub_tier"]== 1]["is_delayed"] == 1).sum()
    tier2_delay = (data[data["hub_tier"]== 2]["is_delayed"] == 1).sum()
    tier3_delay = (data[data["hub_tier"]== 3]["is_delayed"] == 1).sum()

    db1_delay = (data[data["distance_km"] <= 200]["is_delayed"] == 1).sum()
    db2_delay = (data[(data["distance_km"] > 200) & (data["distance_km"] <= 400)]["is_delayed"] == 1).sum()
    db3_delay = (data[(data["distance_km"] > 400) & (data["distance_km"] <= 600)]["is_delayed"] == 1).sum()
    db4_delay = (data[data["distance_km"] > 600]["is_delayed"] == 1).sum()

    df_delay = pd.DataFrame({
        "dimension":["service_type","service_type","service_type","hub_tier","hub_tier","hub_tier","distance_band","distance_band","distance_band","distance_band"],
        "group":["Economy","Express","Standard",1,2,3,"0-200","201-400","401-600","601+"],
        "shipments":[econ_total, exp_total, std_total, hub_tier1, hub_tier2, hub_tier3, dist_band1, dist_band2, dist_band3, dist_band4],
        "delayed":[econ_delay, exp_delay, std_delay, tier1_delay, tier2_delay, tier3_delay, db1_delay, db2_delay, db3_delay, db4_delay],
        "delay_rate":[econ_delay/econ_total, exp_delay/exp_total, std_delay/std_total, tier1_delay/hub_tier1, tier2_delay/hub_tier2, tier3_delay/hub_tier3, db1_delay/dist_band1, db2_delay/dist_band2, db3_delay/dist_band3, db4_delay/dist_band4],
    }).round(4)

    df_delay.to_csv('./outputs/part_a_delay_by_group.csv', index=False)

def corr_matrix():
    df = pd.read_csv("./data/modelling_table.csv")
    dhph = df["delivery_hours"].corr(df["promised_hours"])
    dhdk = df["delivery_hours"].corr(df["distance_km"])
    deldk = df["is_delayed"].corr(df["distance_km"])

    df_new = pd.DataFrame({
        "Correlation_type":["corr(delivery_hours, promised_hours)", "corr(delivery_hours, distance_km)", "corr(is_delayed, distance_km)"],
        "Values":[dhph, dhdk, deldk]
    }).round(4)

    df_new.to_csv("./outputs/part_a_correlations.csv", index=False)
    

main()
checks()
delay_evidence()
corr_matrix()