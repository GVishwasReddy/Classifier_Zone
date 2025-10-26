import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings("ignore")
import segmentation_models as sm
sm.set_framework('tf.keras')

DATA_DIR = 'data/Semantic segmentation dataset'

metadata_df = pd.read_csv(f'{DATA_DIR}/metadata_processed.csv')
train_df = metadata_df[metadata_df['split']=='train']
test_df = metadata_df[metadata_df['split']=='test']

valid_df = train_df.sample(frac=0.1, random_state=42)
train_df = train_df.drop(valid_df.index)

# Augmentations
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
    ], p=0.8),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8)
])

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, df, root_dir, batch_size=16, target_size=(256, 256), n_classes=6, augmentation=None):
        self.df = df
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_classes = n_classes
        self.augmentation = augmentation

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        masks = []

        for i, row in batch_df.iterrows():
            img_path = os.path.join(self.root_dir, row['sat_image_path'])
            mask_path = os.path.join(self.root_dir, row['mask_path'])
            
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

            if self.augmentation:
                augmented = self.augmentation(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            img = tf.image.resize(img, self.target_size)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.image.resize(mask, self.target_size, method='nearest')
            mask = tf.squeeze(mask, axis=-1)

            images.append(img)
            
            # One-hot encode the mask
            mask_one_hot = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)
            masks.append(mask_one_hot)

        return np.array(images), np.array(masks)
    
def get_dataset(df, root_dir, batch_size=16, target_size=(256, 256), n_classes=6, augmentation=None, shuffle=True):
    dataset = CustomDataset(df, root_dir, batch_size, target_size, n_classes, augmentation)
    return dataset

train_dataset = get_dataset(train_df, DATA_DIR, augmentation=augmentation)
valid_dataset = get_dataset(valid_df, DATA_DIR, shuffle=False)
test_dataset = get_dataset(test_df, DATA_DIR, shuffle=False)

def get_model(name, n_classes=6, input_shape=(256, 256, 3), backbone='efficientnetb0', encoder_weights='imagenet', activation='softmax'):
    model = sm.Unet(backbone, classes=n_classes, activation=activation, input_shape=input_shape, encoder_weights=encoder_weights)
    return model

model = get_model('Unet')

# Learning rate scheduler
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + focal_loss

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=total_loss, metrics=[sm.metrics.iou_score])

history = model.fit(
    train_dataset, 
    validation_data=valid_dataset, 
    epochs=50, 
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True),
        lr_schedule
    ]
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model IOU Score')
plt.ylabel('IOU Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

plt.show()

results = model.evaluate(test_dataset)

print(f'Test IOU Score: {results[1]}')
print(f'Test Loss: {results[0]}')

def plot_predictions(model, dataset):
    n_images = 5
    
    images, masks = dataset[0]
    preds = model.predict(images)
    
    plt.figure(figsize=(20, 10))
    
    for i in range(n_images):
        plt.subplot(3, n_images, i+1)
        plt.imshow(images[i])
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(3, n_images, i+1+n_images)
        plt.imshow(np.argmax(masks[i], axis=-1), cmap='jet')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(3, n_images, i+1+2*n_images)
        plt.imshow(np.argmax(preds[i], axis=-1), cmap='jet')
        plt.title('Prediction')
        plt.axis('off')
        
    plt.show()

plot_predictions(model, test_dataset)

