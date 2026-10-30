import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    # 1. Drop exact duplicate rows.
    raw_new = raw.drop_duplicates()

    # 2. Service type strip and title-ized
    raw_new["service_type"] = raw_new["service_type"].str.strip().str.title()

    # 3. payment mode fillna 'unknown'
    raw_new["payment_mode"] = raw_new["payment_mode"].fillna("Unknown")

    # 4. declared value conversion to int
    raw_new["declared_value_inr"] = raw_new["declared_value"].str.replace(',','').str.strip('₹')
    raw_new["declared_value_inr"] = pd.to_numeric(raw_new["declared_value_inr"], errors='coerce')

    # drop declared_value str column
    raw_new = raw_new.drop(columns="declared_value")

    # 5. dispatch time convert to date time
    raw_new["dispatch_time"] = pd.to_datetime(raw_new["dispatch_time"])

    # 6. parcel weight over 100 = missing
    raw_new.loc[raw_new["parcel_weight_kg"]>100, "parcel_weight_kg"] = None

    # 7. fill missing distance_km, parcel_weight_kg and declared_value_inr with that column's median.

    # calculate 3 medians:
    med_dis = raw_new["distance_km"].median()
    med_weight = raw_new["parcel_weight_kg"].median()
    med_inr = raw_new["declared_value_inr"].median()

    # use fillna for 3 medians:
    raw_new["distance_km"] = raw_new["distance_km"].fillna(round(med_dis,2))
    raw_new["parcel_weight_kg"] = raw_new["parcel_weight_kg"].fillna(med_weight)
    raw_new["declared_value_inr"] = raw_new["declared_value_inr"].fillna(med_inr)

    # medians dictionary
    medians = {"distance_km":med_dis,
               "parcel_weight_kg":med_weight,
               "declared_value_inr":med_inr}

    return(raw_new, medians)

def build_features(clean, hubs):
    # left merge clean + hubs on hub_id
    merged_df = pd.merge(clean, hubs, on="hub_id", how="left")

    # map promised_hours with PROMISED_HOURS
    merged_df["promised_hours"] = merged_df["service_type"].map(PROMISED_HOURS).astype(int)

    # implement is_cod, if payment_mode = COD use 1, else 0
    merged_df["is_cod"] = (merged_df["payment_mode"] == "COD").astype(int)

    # implement is_weekend, if dispatch_day = sunday or saturday use 1, else 0
    merged_df["is_weekend"] = (merged_df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    # implement dispatch_hour, check dispatch_hour use the hour on it
    merged_df["dispatch_hour"] = merged_df["dispatch_time"].dt.hour

    # implement log_declared_value which is natural log of declared_value_inr
    merged_df["log_declared_value"] = round(np.log(merged_df["declared_value_inr"]), 4)

    return(merged_df)


if __name__ == "__main__":
    raw = pd.read_csv("../data/raw_shipments.csv")
    hubs = pd.read_csv("../data/hubs.csv")
    clean, medians = clean_shipments(raw)
    final = build_features(clean=clean, hubs=hubs)
    print(final)