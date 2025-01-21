import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time

# Record start time
start = time.time()

# Load the DataFrame
df = pd.read_csv("predictions/prediction_X_all_estriatum_points_41.tsv", sep="\t")

# Initialize the zero array
zero_array = np.zeros((340, 673, 552))

for i in range(340):
    for j in range(673):
        for t in range(1):
            value = df[(df["pxl_row_transformed"] == i) & (df["pxl_column_transformed"] == j) & (df["slice"] == t)]["soft_prob_Striatal neuroblast"]
            if not value.empty:
               zero_array[i,j,t] =  value.iloc[0]

# Update the zero_array with the results

# Record end time
end = time.time()

# Print the execution time
print("The time of execution of above program is :", (end - start), "s")