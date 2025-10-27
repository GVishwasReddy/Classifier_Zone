import os
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = 'segmentation_project/data'
metadata_df = pd.read_csv(f'{DATA_DIR}/all_metadata.csv')

mask_files = metadata_df['mask_path'].tolist()

all_unique_values = set()
for f in mask_files:
    mask_path = os.path.join(DATA_DIR, f)
    mask = np.array(Image.open(mask_path))
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    unique_values = np.unique(mask)
    all_unique_values.update(unique_values)

sorted_unique_values = sorted(list(all_unique_values))
mapping = {val: i for i, val in enumerate(sorted_unique_values)}

print("All unique values in masks:", sorted_unique_values)
print("Mapping to class indices:", mapping)
