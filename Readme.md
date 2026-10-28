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