# Semantic Segmentation Based Optimal Drone Landing Zone

This project focuses on semantic segmentation of aerial imagery using a U-Net model with a ResNet50 backbone. The goal is to accurately identify and classify different objects and regions within aerial images.

## Project Structure

The project is organized into the following directories:

*   `data/`: Contains all the datasets used in the project. This directory is ignored by Git.
*   `src/`: Contains all the Python source code, including the training script, data preprocessing scripts, and utility functions.
*   `models/`: Stores the trained model weights (`model.h5`). This directory is ignored by Git.
*   `notebooks/`: Contains Jupyter notebooks for experimentation and analysis. This directory is ignored by Git.
*   `tools/`: Contains external tools or utilities used in the project (e.g., PlotNeuralNet).
*   `.venv/`: Python virtual environment for managing dependencies. This directory is ignored by Git.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GVishwasReddy/Classifier_Zone.git
    cd Classifier_Zone
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the semantic segmentation model:

1.  Ensure your virtual environment is activated (`source .venv/bin/activate`).
2.  Run the training script:
    ```bash
    python src/train.py
    ```
    *Note: The training process can take a significant amount of time. The script will save the best model weights to `models/model.h5`.*

### Running Predictions with a Trained Model

If you already have a trained model (`models/model.h5`) and want to run predictions without retraining:

1.  Ensure your virtual environment is activated (`source .venv/bin/activate`).
2.  Open `src/train.py` in a text editor.
3.  **Comment out** the training block (the `history = model.fit(...)` section).
4.  **Uncomment** the line `model.load_weights('models/model.h5')`.
5.  Run the script:
    ```bash
    python src/train.py
    ```
    *The script will load the saved model and display predictions on a few random images from the test set.*

## Datasets

This project utilizes several semantic segmentation datasets:

*   **Semantic Drone Dataset:** [Kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)
*   **Urban Segmentation ISPRS:** [ISPRS Website](https://www.isprs.org/)
*   **OpenEarthMap (Global Land Cover Mapping):** [OpenEarthMap Website](https://open-earth-map.org)
*   **Swiss Drone and Okutama Drone Datasets:** [Okutama Segmentation](https://okutama-segmentation.org)
*   **Semantic Segmentation of Aerial Imagery:** [Kaggle](https://www.kaggle.com/datasets/bulentsiyah/semantic-segmentation-of-aerial-imagery)

## Model Details

*   **Architecture:** U-Net with ResNet50 backbone.
*   **Input Size:** 384x384 pixels.
*   **Loss Function:** A combination of Categorical Crossentropy and Dice Loss.
*   **Classes:** 12 classes for segmentation.

## Contributing

(Add contributing guidelines if applicable)

## License

(Add license information if applicable)
