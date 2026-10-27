import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

def main():
    hubs = pd.read_csv("./data/hubs.csv")
    raw = pd.read_csv("./data/raw_shipments.csv")

    clean, medians = clean_shipments(raw)
    final = build_features(clean, hubs)

    final["is_delayed"] = (final["delivery_hours"]>final["promised_hours"]).astype(int)

    final.to_csv('./data/modelling_table.csv', index=False)

main()