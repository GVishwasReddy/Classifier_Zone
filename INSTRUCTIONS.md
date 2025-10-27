
# Instructions for Semantic Segmentation Project

This project performs semantic segmentation on drone imagery using a U-Net model with an EfficientNet backbone.

## 1. Project Structure

- `semantic-drone-segmentation-unet-resnet34.ipynb`: The main Jupyter notebook for running predictions and visualizing results.
- `src/`: Contains the Python source code for data processing and model training.
  - `semantic_segmentation.py`: The main script for training the model.
  - `create_metadata.py`: A script to generate the metadata file for all datasets.
  - `preprocess_masks.py`: A script to preprocess masks for the "Semantic segmentation dataset".
- `data/`: Contains the datasets.
- `requirements.txt`: A list of Python dependencies.
- `model.h5`: The trained model.

## 2. Setup

### 2.1. Install Dependencies

It is recommended to use a virtual environment to avoid conflicts with system packages.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

For Apple Silicon Macs, make sure you have `tensorflow-metal` installed to use the GPU.

### 2.2. Prepare the Data

The project is pre-configured to use the datasets in the `data` directory. If you want to add your own dataset, follow these steps:

1.  Create a new folder for your dataset inside the `data` directory.
2.  Inside your dataset folder, create two sub-folders: `images` and `masks`.
3.  Place your satellite/drone images in the `images` folder and the corresponding segmentation masks in the `masks` folder.
4.  Run the `create_metadata.py` script to update the `all_metadata.csv` file:

    ```bash
    python src/create_metadata.py
    ```

## 3. Training

To train the model, run the `semantic_segmentation.py` script:

```bash
python src/semantic_segmentation.py
```

The script will:

1.  Load the data using the `all_metadata.csv` file.
2.  Preprocess the images and masks.
3.  Train the U-Net model.
4.  Save the best model to `model.h5`.

## 4. Prediction

To run predictions and visualize the results, open and run the `semantic-drone-segmentation-unet-resnet34.ipynb` notebook in a Jupyter environment.

### 4.1. Running on an Image

To run the model on a new image:

1.  Place your image in the `input` directory (create it if it doesn't exist).
2.  In the notebook, modify the `IMAGE_PATH` variable to point to your image.
3.  Run the notebook cells to see the segmentation map.

### 4.2. Running on a Video

To run the model on a video:

1.  Place your video in the `input` directory.
2.  The notebook is not pre-configured to run on videos. You will need to add code to:
    -   Read the video frame by frame using a library like OpenCV (`cv2`).
    -   For each frame, preprocess it and feed it to the model for prediction.
    -   Display the resulting segmentation map on the frame.

Here is a sample code snippet to get you started with video processing:

```python
import cv2

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Open the video file
cap = cv2.VideoCapture('input/your_video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Preprocess the frame (resize, normalize, etc.)
        # ...

        # Get the prediction from the model
        prediction = model.predict(preprocessed_frame)

        # Post-process the prediction and display it
        # ...

        cv2.imshow('Frame', displayed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
```
