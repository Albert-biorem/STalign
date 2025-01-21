# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:17:39 2025

@author: Albert
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
 
# record start time
start = time.time()
df = pd.read_csv("predictions/prediction_X_all_estriatum_points_41.tsv", sep="\t")
# Initialize the zero array
zero_array = np.zeros((340, 673, 552))

# Define the function to process each element
def process_element(i, j, t):
    value = df[(df["pxl_row_transformed"] == i) & (df["pxl_column_transformed"] == j) & (df["slice"] == t)]["soft_prob_Striatal neuroblast"]
    if not value.empty:
        zero_array[i, j, t] = value.iloc[0]


# Use joblib to parallelize the processing
Parallel(n_jobs=-1)(delayed(process_element)(i, j, t) for i in range(340) for j in range(673) for t in range(2))
end = time.time()
 
# print the difference between start 
# and end time in milli. secs
print("The time of execution of above program is :",
      (end-start) , "s")