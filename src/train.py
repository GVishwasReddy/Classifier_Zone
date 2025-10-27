import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = 'data'

metadata_df = pd.read_csv(f'{DATA_DIR}/all_metadata.csv')

train_df, test_df = train_test_split(metadata_df, test_size=0.1, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 60: 6, 110: 7, 132: 8, 155: 9, 226: 10, 254: 11}

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, df, root_dir, batch_size=16, target_size=(384, 384), n_classes=12, mapping=None):
        self.df = df
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_classes = n_classes
        self.mapping = mapping

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

            if len(mask.shape) == 3:
                mask = mask[..., 0]

            # Apply mapping
            new_mask = np.zeros_like(mask)
            for k, v in self.mapping.items():
                new_mask[mask == k] = v
            mask = new_mask

            img = tf.image.resize(img, self.target_size)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.image.resize(mask, self.target_size, method='nearest')
            mask = tf.squeeze(mask, axis=-1)

            images.append(img)
            
            # One-hot encode the mask
            mask_one_hot = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)
            masks.append(mask_one_hot)

        return np.array(images), np.array(masks)

def get_dataset(df, root_dir, batch_size=16, target_size=(384, 384), n_classes=12, mapping=None, shuffle=True):
    dataset = CustomDataset(df, root_dir, batch_size, target_size, n_classes, mapping)
    return dataset

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def total_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return loss + dice

def get_model(name, n_classes=12, input_shape=(384, 384, 3), backbone='resnet50', encoder_weights='imagenet', activation='softmax'):
    resnet50 = tf.keras.applications.ResNet50(weights=encoder_weights, include_top=False, input_shape=input_shape)

    s1 = resnet50.get_layer("conv1_relu").output
    s2 = resnet50.get_layer("conv2_block3_out").output
    s3 = resnet50.get_layer("conv3_block4_out").output
    s4 = resnet50.get_layer("conv4_block6_out").output

    b1 = resnet50.get_layer("conv5_block3_out").output

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    x = layers.UpSampling2D((2, 2))(d4)
    outputs = layers.Conv2D(n_classes, 1, padding="same", activation=activation)(x)

    model = tf.keras.Model(inputs=resnet50.input, outputs=outputs, name="U-Net")
    return model

model = get_model('Unet', input_shape=(384, 384, 3))

model.load_weights('models/model.h5')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=total_loss, metrics=['accuracy', dice_coefficient, tf.keras.metrics.MeanIoU(num_classes=12)])

def plot_predictions(model, dataset, title):
    n_images = 5
    
    images, masks = dataset[0]
    preds = model.predict(images)
    
    plt.figure(figsize=(20, 12))
    plt.suptitle(title, fontsize=16)
    
    for i in range(n_images):
        plt.subplot(3, n_images, i+1)
        plt.imshow(images[i].astype('uint8'))
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

for i in range(3):
    random_test_df = test_df.sample(5)
    random_test_dataset = get_dataset(random_test_df, DATA_DIR, mapping=mapping, shuffle=False)
    plot_predictions(model, random_test_dataset, f"Random Set {i+1}")