import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

store_data = pd.read_csv("store_data.csv", header=None)
#print(store_data.head())

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.009, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

for item in association_results:

    print(item, "\n")