import os
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = 'segmentation_project/data'
metadata_df = pd.read_csv(f'{DATA_DIR}/all_metadata.csv')

mask_files = metadata_df['mask_path'].tolist()

for i in range(10): # inspect the first 10 masks
    mask_path = os.path.join(DATA_DIR, mask_files[i])
    mask = np.array(Image.open(mask_path))
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    print(f"Mask: {mask_files[i]}, Unique values: {np.unique(mask)}")
