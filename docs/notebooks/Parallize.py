import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
import SimpleITK as sitk

# Record start time
start = time.time()

# Load the DataFrame
df = pd.read_csv("predictions/prediction_X_all_estriatum_points_41.tsv", sep="\t")


df = df[["pxl_row_transformed", "pxl_column_transformed", "slice", "hard_pred"]]
replace_dict ={ 'NPC':1, 'NSC':2, 'Cortical neuron':3, 'Embryonic dSPN':4,
       'Striatal neuroblast':5}
df = df.replace(replace_dict)
# Initialize the zero array
zero_array = np.zeros((340, 673, 552))

# Define the function to process each element
def process_element(i, j, t):
    value = df[(df["pxl_row_transformed"] == i) & (df["pxl_column_transformed"] == j) & (df["slice"] == t)]["hard_pred"]
    if not value.empty:
        return (i, j, t, value.iloc[0])
    return (i, j, t, 0)

# Prepare the list of parameters for each element
params_list = [(i, j, t) for i in range(107,212) for j in range(105,157) for t in range(184,295)]

# Use joblib to parallelize the processing
results = Parallel(n_jobs=-1, batch_size='auto')(delayed(process_element)(params[0], params[1], params[2]) for params in params_list)

# Update the zero_array with the results
for i, j, t, val in results:
    zero_array[i, j, t] = val
# Record end time
end = time.time()

# Print the execution time
print("The time of execution of above program is :", (end - start), "s")
image = sitk.GetImageFromArray(zero_array)
sitk.WriteImage(image, "visualizations/hard_pred1.mhd")