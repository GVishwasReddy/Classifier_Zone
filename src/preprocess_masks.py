import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = 'data/Semantic segmentation dataset'

metadata_df = pd.read_csv(f'{DATA_DIR}/metadata.csv')

color_map = {
    (80, 227, 194): 0,  # Water
    (245, 166, 35): 1,   # Land
    (222, 89, 127): 2,   # Road
    (208, 2, 27): 3,      # Building
    (65, 117, 5): 4,     # Vegetation
    (155, 155, 155): 5, # Unlabeled
}

def preprocess_mask(mask_path):
    mask = Image.open(mask_path)
    mask = mask.resize((256, 256))
    mask = np.array(mask)

    if len(mask.shape) == 2:
        mask = np.stack([mask, mask, mask], axis=-1)

    if mask.shape[-1] == 4:
        mask = mask[..., :3]

    mask_indices = np.zeros((256, 256), dtype=np.uint8)
    for color, index in color_map.items():
        mask_indices[np.all(mask == np.array(color), axis=-1)] = index

    return Image.fromarray(mask_indices)

for i, row in metadata_df.iterrows():
    mask_path = os.path.join(DATA_DIR, row['mask_path'])
    new_mask_path = mask_path.replace('.png', '_processed.png')
    
    if not os.path.exists(os.path.dirname(new_mask_path)):
        os.makedirs(os.path.dirname(new_mask_path))

    processed_mask = preprocess_mask(mask_path)
    processed_mask.save(new_mask_path)

    metadata_df.loc[i, 'mask_path'] = row['mask_path'].replace('.png', '_processed.png')

metadata_df.to_csv(f'{DATA_DIR}/metadata_processed.csv', index=False)
