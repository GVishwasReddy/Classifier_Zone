
import os
import pandas as pd
import glob

def create_metadata():
    datasets = [
        {
            "name": "global-land-cover-mapping-openearthmap",
            "image_dir": "data/global-land-cover-mapping-openearthmap/images",
            "mask_dir": "data/global-land-cover-mapping-openearthmap/label",
        },
        {
            "name": "semantic-drone-dataset",
            "image_dir": "data/semantic-drone-dataset/original_images",
            "mask_dir": "data/semantic-drone-dataset/label_images_semantic",
        },
        {
            "name": "swiss-drone-and-okutama-drone-datasets",
            "image_dir": "data/swiss-drone-and-okutama-drone-datasets/images",
            "mask_dir": "data/swiss-drone-and-okutama-drone-datasets/ground_truth",
        },
        {
            "name": "urban-segmentation-isprs-potsdam",
            "image_dir": "data/urban-segmentation-isprs/Potsdam/Images",
            "mask_dir": "data/urban-segmentation-isprs/Potsdam/Labels",
        },
        {
            "name": "urban-segmentation-isprs-vaihingen",
            "image_dir": "data/urban-segmentation-isprs/Vaihingen/Images",
            "mask_dir": "data/urban-segmentation-isprs/Vaihingen/Labels",
        },
    ]

    all_df = []

    # For datasets with explicit image and mask directories
    for dataset in datasets:
        image_paths = sorted(glob.glob(os.path.join(dataset["image_dir"], "*.png")))
        mask_paths = sorted(glob.glob(os.path.join(dataset["mask_dir"], "*.png")))
        if len(image_paths) == len(mask_paths):
            df = pd.DataFrame({
                "sat_image_path": [p.replace('data/', '') for p in image_paths],
                "mask_path": [p.replace('data/', '') for p in mask_paths],
            })
            all_df.append(df)
        else:
            print(f"Warning: Mismatch in number of images and masks for {dataset['name']}")

    # For "Semantic segmentation dataset"
    semantic_df = pd.read_csv("data/Semantic segmentation dataset/metadata.csv")
    semantic_df["sat_image_path"] = "Semantic segmentation dataset/" + semantic_df["sat_image_path"]
    semantic_df["mask_path"] = "Semantic segmentation dataset/" + semantic_df["mask_path"]
    all_df.append(semantic_df[["sat_image_path", "mask_path"]])
    
    combined_df = pd.concat(all_df, ignore_index=True)
    
    # Add a split column (80% train, 20% test)
    train_df = combined_df.sample(frac=0.8, random_state=42)
    test_df = combined_df.drop(train_df.index)
    train_df["split"] = "train"
    test_df["split"] = "test"
    
    final_df = pd.concat([train_df, test_df], ignore_index=True)

    final_df.to_csv("data/all_metadata.csv", index=False)
    print("Created data/all_metadata.csv")

if __name__ == '__main__':
    create_metadata()
