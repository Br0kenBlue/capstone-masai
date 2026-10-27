import numpy as np
import pandas as pd

def clean_shipments(raw, medians=None):

    return(raw.head())




if __name__ == "__main__":
    raw = pd.read_csv("../data/raw_shipments.csv")
    print(clean_shipments(raw))