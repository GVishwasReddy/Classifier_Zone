
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
import segmentation_models as sm
sm.set_framework('tf.keras')
import warnings
warnings.filterwarnings("ignore")

def rgb2ohe(mask, colors):
    for i, color in enumerate(colors):
        boolean_mask = tf.reduce_all(tf.equal(mask, tf.constant(color, dtype=tf.uint8)), axis=-1)
        mask_i = tf.where(boolean_mask, 1., 0.)
        mask_i = tf.expand_dims(mask_i, axis=-1)
        if i == 0:
            mask_ohe = mask_i
        else:
            mask_ohe = tf.concat([mask_ohe, mask_i], axis=2)
    return mask_ohe

def get_edges_from_mask(mask_rgb, rgb_format=True, lw=1):
    mask_rgb = tf.cast(mask_rgb, dtype=tf.int32)
    image_edge = tf.zeros(shape=[256, 256, 1], dtype=tf.uint8)
    for i in range(lw):
        if i == 0:
            dy, dx = tf.image.image_gradients(tf.cast(tf.expand_dims(mask_rgb, axis=0), dtype=tf.int32))
        else:
            dy, dx = tf.image.image_gradients(tf.expand_dims(image_edge_, axis=0))

        image_edge_ = tf.cast(tf.squeeze(dy**2 + dx**2), dtype=tf.int32)
        image_edge_ = tf.where(image_edge_ != 0, [255, 255, 255], [0, 0, 0])
        image_edge += tf.cast(image_edge_, dtype=tf.uint8)

    if not rgb_format:
        image_edge = tf.reduce_max(image_edge, axis=-1, keepdims=True)
        image_edge = tf.where(image_edge != 0, 1, 0)
    return image_edge

def data_transformation(image, mask):
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.9, upper=1)
    image = tf.image.random_hue(image, max_delta=0.09)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=75, max_jpeg_quality=100)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    aux = tf.concat([tf.cast(image, dtype=tf.float32), tf.cast(mask, dtype=tf.float32)], -1)

    aux = tf.image.random_flip_left_right(aux)
    aux = tf.image.random_flip_up_down(aux)

    return tf.cast(aux[..., :3], tf.uint8), tf.cast(aux[..., 3:], tf.uint8)

def data_augmentation(image, mask_rgb, edge=False, train=True):
    image = tf.cast(image, dtype=tf.uint8)
    mask_rgb = tf.cast(mask_rgb, dtype=tf.uint8)

    if train == True:
        image, mask_rgb = data_transformation(image, mask_rgb)

    image = tf.reshape(image, shape=(256, 256, 3))
    mask_rgb = tf.reshape(mask_rgb, shape=(256, 256, 3))
    mask_ohe = rgb2ohe(mask_rgb, Labels().get_colors())

    image = tf.cast(image, dtype=tf.float32)/255
    mask_ohe = tf.cast(mask_ohe, dtype=tf.float32)

    if edge == True:
        edges = get_edges_from_mask(mask_rgb, rgb_format=False, lw=3)
        edges = tf.cast(edges, dtype=tf.float32)
        return image, {"segmentation": mask_ohe, "edge_detection_encoder": edges, "edge_detection_decoder": edges}
    else:
        return image, mask_ohe

class Labels:
    def __init__(self):
        self.classes = ["Water", "Land", "Road", "Building", "Vegetation", "Unlabeled"]
        self.colors = [[80, 227, 194], [245, 166, 35], [222, 89, 127], [208, 2, 27], [65, 117, 5], [155, 155, 155]]

    def get_classes(self):
        return self.classes

    def get_colors(self):
        return self.colors

    def get_class(self, color):
        for i, c in enumerate(self.colors):
            if np.array_equal(c, color):
                return self.classes[i]
        return None

    def get_color(self, class_name):
        for i, c in enumerate(self.classes):
            if c == class_name:
                return self.colors[i]
        return None


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, df, root_dir, batch_size=16, target_size=(256, 256), n_classes=6, augmentation=None, training=True):
        self.df = df
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.training = training

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

            img = tf.image.resize(img, self.target_size)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.image.resize(mask, self.target_size, method='nearest')
            mask = tf.image.grayscale_to_rgb(mask)
            img, mask_dict = data_augmentation(img, mask, edge=True, train=self.training)
            
            images.append(img)
            masks.append(mask_dict)

        return np.array(images), {"segmentation": np.array([m["segmentation"] for m in masks]), "edge_detection_encoder": np.array([m["edge_detection_encoder"] for m in masks]), "edge_detection_decoder": np.array([m["edge_detection_decoder"] for m in masks])}



@tf.keras.utils.register_keras_serializable()
class Encoder(keras.Model):
    def __init__(self, size_image=512, backbone=None, training_up_to_layer=35, mini_blocks=3, dropout=None, bn=False, initializer='he_normal', regularizer=None, name="Encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = dropout
        self.state_backbone = backbone
        self.training_up_to_layer = training_up_to_layer
        self.bn = bn
        self.mini_blocks = mini_blocks
        self.size_image = size_image

        if self.state_backbone:
            self.preprocess_input, self.backbone = self.get_backbone()
        else:
            self.convs_block1, self.relus_block1, self.bns_block1, self.drop1_block1, self.mxp_block1 = self.get_block(filters=13, num_block=1)
            self.convs_block2, self.relus_block2, self.bns_block2, self.drop1_block2, self.mxp_block2 = self.get_block(filters=32, num_block=2)
            self.convs_block3, self.relus_block3, self.bns_block3, self.drop1_block3, self.mxp_block3 = self.get_block(filters=64, num_block=3)
            self.convs_block4, self.relus_block4, self.bns_block4, self.drop1_block4, self.mxp_block4 = self.get_block(filters=128, num_block=4)
            self.convs_block5, self.relus_block5, self.bns_block5, self.drop1_block5, self.mxp_block5 = self.get_block(filters=256, num_block=5)

        self.convs_block6, self.relus_block6, self.bns_block6, self.drop1_block6, self.mxp_block6 = self.get_block(filters=256, num_block=6)

    def get_block(self, filters=None, div=None, num_block=None):
        convs = []
        relus = []
        bns = []

        n=0
        for i in range(self.mini_blocks):
            convs.append(self.conv2d(filters=filters, name=f"en_block{num_block}_conv{n}"))
            if self.bn:
                bns.append(layers.BatchNormalization(name=f"en_block{num_block}_bn{n}"))
            relus.append(layers.ReLU(name=f"en_block{num_block}_relu{n}"))
            n += 1

            convs.append(self.conv2d(filters=filters, name=f"en_block{num_block}_conv{n}"))
            if self.bn:
                bns.append(layers.BatchNormalization(name=f"en_block{num_block}_bn{n}"))
            relus.append(layers.ReLU(name=f"en_block{num_block}_relu{n}"))
            n += 1

        mxp = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{num_block}_mxp1")
        drop = layers.Dropout(self.dropout, name=f"en_block{num_block}_drop1")
        return convs, relus, bns, drop, mxp

    def conv2d(self, filters, name):
        return layers.Conv2D(filters=filters, kernel_size=(3, 3), activation=None, padding='same', strides=(1, 1), kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, data_format="channels_last", name=name)

    def conv_block(self, _input, convs, relus, bns, drop, mxp):
        x = _input
        n=0

        for i in range(self.mini_blocks):
            skip = x = convs[n](x)
            x = relus[n](x)
            x = bns[n](x) if self.bn else x
            n += 1

            x = convs[n](x)
            x = relus[n](x)
            x = bns[n](x) if self.bn else x

            n += 1

            x = layers.Add()([skip, x])

        x = layers.Concatenate()([_input, x])

        block_output = x
        x = mxp(x)
        x = drop(x)
        return x, block_output

    def get_backbone(self):
        from classification_models.tfkeras import Classifiers
        ResNet34, preprocess_input = Classifiers.get('resnet34')
        backbone = ResNet34(input_shape=[None, None, 3], weights='imagenet')

        for i, layer in enumerate(backbone.layers):
            if (i <= self.training_up_to_layer):
                layer.trainable = False

        block1_output, block2_output, block3_output, block4_output, block5_output, block6_output = backbone.layers[1].output, backbone.layers[5].output, backbone.layers[37].output, backbone.layers[74].output, backbone.layers[129].output, backbone.layers[157].output

        return preprocess_input, keras.Model(inputs=[backbone.inputs], outputs=[block1_output, block2_output, block3_output, block4_output, block5_output, block6_output])

    def call(self, inputs, training=False):
        if self.state_backbone:
            x = self.preprocess_input(inputs)
            block1_output, block2_output, block3_output, block4_output, block5_output, block6_output = self.backbone(x, training=training)
            x = self.mxp_block6(block6_output)
            encoder_output = self.drop1_block6(x)
        else:
            x, block1_output = self.conv_block(_input=inputs, convs=self.convs_block1, relus=self.relus_block1, bns=self.bns_block1, mxp=self.mxp_block1, drop=self.drop1_block1)
            x, block2_output = self.conv_block(_input=x, convs=self.convs_block2, relus=self.relus_block2, bns=self.bns_block2, mxp=self.mxp_block2, drop=self.drop1_block2)
            x, block3_output = self.conv_block(_input=x, convs=self.convs_block3, relus=self.relus_block3, bns=self.bns_block3, mxp=self.mxp_block3, drop=self.drop1_block3)
            x, block4_output = self.conv_block(_input=x, convs=self.convs_block4, relus=self.relus_block4, bns=self.bns_block4, mxp=self.mxp_block4, drop=self.drop1_block4)
            x, block5_output = self.conv_block(_input=x, convs=self.convs_block5, relus=self.relus_block5, bns=self.bns_block5, mxp=self.mxp_block5, drop=self.drop1_block5)
            encoder_output, block6_output = self.conv_block(_input=x, convs=self.convs_block6, relus=self.relus_block6, bns=self.bns_block6, mxp=self.mxp_block6, drop=self.drop1_block6)

        return encoder_output, [block1_output, block2_output, block3_output, block4_output, block5_output, block6_output]

@tf.keras.utils.register_keras_serializable()
class Bottleneck(keras.Model):
    def __init__(self, max_filters=128, backbone=False, dropout=None, bn=False, initializer='he_normal', regularizer=None, name="Bottleneck", **kwargs):
        super().__init__(name=name, **kwargs)
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = dropout
        self.bn = bn

        self.bn1 = layers.BatchNormalization(name="btn_bn1")
        self.bn2 = layers.BatchNormalization(name="btn_bn2")
        self.btn_conv1 = self.conv2d(filters=max_filters, name="btn_conv1")
        self.btn_conv2 = self.conv2d(filters=max_filters, name="btn_conv2")
        self.relu1 = layers.ReLU(name="btn_relu1")
        self.relu2 = layers.ReLU(name="btn_relu2")
        self.bn3 = layers.BatchNormalization(name="btn_bn3")
        self.bn4 = layers.BatchNormalization(name="btn_bn4")
        self.btn_conv3 = self.conv2d(filters=max_filters*2, name="btn_conv3")
        self.btn_conv4 = self.conv2d(filters=512 if backbone else 752, name="btn_conv4")
        self.relu3 = layers.ReLU(name="btn_relu3")
        self.relu4 = layers.ReLU(name="btn_relu4")

        self.btn_drop1 = layers.Dropout(self.dropout, name="btn_drop1")
        self.btn_upconv1 = self.conv2dtranspose(filters=512 if backbone else 752, name="btn_upconv1")

    def conv2d(self, filters, name):
        return layers.Conv2D(filters=filters, kernel_size=(3, 3), activation=None, padding='same', strides=(1, 1), kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, data_format="channels_last", name=name)

    def conv2dtranspose(self, filters, name):
        return layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), activation="relu", kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, data_format="channels_last", name=name)

    def call(self, inputs, training=False):
        skip = x = self.btn_conv1(inputs)
        x = self.relu1(x)
        x = self.bn1(x) if self.bn else x
        x = self.btn_conv2(x)
        x = self.relu2(x)
        x = self.bn2(x) if self.bn else x
        x = layers.Add()([skip, x])

        x = self.btn_conv3(x)
        x = self.relu3(x)
        x = self.bn3(x) if self.bn else x
        x = self.btn_conv4(x)
        x = self.relu4(x)
        x = self.bn4(x) if self.bn else x
        x = layers.Add()([inputs, x])

        x = self.btn_upconv1(x)
        bottleneck_output = self.btn_drop1(x)

        return bottleneck_output

@tf.keras.utils.register_keras_serializable()
class Decoder(keras.Model):
    def __init__(self, mini_blocks=3, backbone=False, dropout=None, bn=False, initializer='he_normal', regularizer=None, name="Decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = dropout
        self.bn = bn
        self.mini_blocks = mini_blocks

        self.convs_block1, self.relus_block1, self.bns_block1, self.drop1_block1, self.upconv1_block1 = self.get_block(filters=512 if backbone else 752, filters_up=256 if backbone else 496, num_block=1)
        self.convs_block2, self.relus_block2, self.bns_block2, self.drop1_block2, self.upconv1_block2 = self.get_block(filters=256 if backbone else 496, filters_up=128 if backbone else 240, num_block=2)
        self.convs_block3, self.relus_block3, self.bns_block3, self.drop1_block3, self.upconv1_block3 = self.get_block(filters=128 if backbone else 240, filters_up=64 if backbone else 112, num_block=3)
        self.convs_block4, self.relus_block4, self.bns_block4, self.drop1_block4, self.upconv1_block4 = self.get_block(filters=64 if backbone else 112, filters_up=64 if backbone else 48, num_block=4)
        self.convs_block5, self.relus_block5, self.bns_block5, self.drop1_block5, self.upconv1_block5 = self.get_block(filters=64 if backbone else 48, filters_up=3 if backbone else 16, num_block=5)
        self.convs_block6, self.relus_block6, self.bns_block6, self.drop1_block6, _ = self.get_block(filters=3 if backbone else 16, filters_up=16 if backbone else 16, num_block=6)
        self.bn_block7 = layers.BatchNormalization(name=f"de_block7_bn1")
        self.conv1_block7 = self.conv2d(16, name="de_block7_conv1")
        self.relu1_block7 = layers.ReLU(name="de_block7_relu1")
        self.conv2_block7 = layers.Conv2D(filters=6, data_format="channels_last", kernel_size=(1, 1), activation="softmax", padding='same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, strides=(1, 1), name="decoder_output")

    def get_block(self, filters, filters_up, num_block):
        convs = []
        bn = []
        relus = []
        n=0

        for i in range(self.mini_blocks):
            convs.append(self.conv2d(filters=filters, name=f"de_block{num_block}_conv{n}"))
            if self.bn:
                bn.append(layers.BatchNormalization(name=f"de_block{num_block}_bn{n}"))
            relus.append(layers.ReLU(name=f"de_block{num_block}_relu{n}"))
            n += 1

            convs.append(self.conv2d(filters=filters, name=f"de_block{num_block}_conv{n}"))
            if self.bn:
                bn.append(layers.BatchNormalization(name=f"de_block{num_block}_bn{n}"))
            relus.append(layers.ReLU(name=f"de_block{num_block}_relu{n}"))
            n += 1

        upconv1 = self.conv2dtranspose(filters=filters_up, name=f"de_block{num_block}_upconv1")
        drop_ = layers.Dropout(self.dropout, name=f"de_block{num_block}_drop1")
        return convs, relus, bn, drop_, upconv1

    def conv2d(self, filters=64, kernel_size=(3, 3), strides=(1, 1), name=None):
        return layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding='same', strides=strides, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, data_format="channels_last", name=name)

    def conv2dtranspose(self, filters=64, kernel_size=(2, 2), strides=(2, 2), name=None):
        return layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, data_format="channels_last", name=name)

    def conv_block(self, _input, skip_connection, convs, relus, bns, drop, upconv1=None):
        x = layers.Add()([skip_connection, _input])
        input_skip = x
        n=0

        for i in range(self.mini_blocks):
            skip = x = convs[n](x)
            x = relus[n](x)
            x = bns[n](x) if self.bn else x
            n += 1

            x = convs[n](x)
            x = relus[n](x)
            x = bns[n](x) if self.bn else x
            n += 1

            x = layers.Add()([skip, x])

        x = layers.Add()([input_skip, x])
        de_block_output = x

        if upconv1:
            x = upconv1(x)

        x = drop(x)

        return x, de_block_output

    def call(self, inputs, skip_connections, training=False):
        x, de_block1_output = self.conv_block(_input=inputs, skip_connection=skip_connections[-1], convs=self.convs_block1, relus=self.relus_block1, bns=self.bns_block1, drop=self.drop1_block1, upconv1=self.upconv1_block1)
        x, de_block2_output = self.conv_block(_input=x, skip_connection=skip_connections[-2], convs=self.convs_block2, relus=self.relus_block2, bns=self.bns_block2, drop=self.drop1_block2, upconv1=self.upconv1_block2)
        x, de_block3_output = self.conv_block(_input=x, skip_connection=skip_connections[-3], convs=self.convs_block3, relus=self.relus_block3, bns=self.bns_block3, drop=self.drop1_block3, upconv1=self.upconv1_block3)
        x, de_block4_output = self.conv_block(_input=x, skip_connection=skip_connections[-4], convs=self.convs_block4, relus=self.relus_block4, bns=self.bns_block4, drop=self.drop1_block4, upconv1=self.upconv1_block4)
        x, de_block5_output = self.conv_block(_input=x, skip_connection=skip_connections[-5], convs=self.convs_block5, relus=self.relus_block5, bns=self.bns_block5, drop=self.drop1_block5, upconv1=self.upconv1_block5)
        _, de_block6_output = self.conv_block(_input=x, skip_connection=skip_connections[-6], convs=self.convs_block6, relus=self.relus_block6, bns=self.bns_block6, drop=self.drop1_block6)

        x = self.conv1_block7(de_block6_output)
        x = self.relu1_block7(x)
        x = self.bn_block7(x) if self.bn else x
        decoder_output = self.conv2_block7(x)

        return decoder_output, [de_block1_output, de_block2_output, de_block3_output, de_block4_output, de_block5_output, de_block6_output]

@tf.keras.utils.register_keras_serializable()
class ERN(keras.Model):
    def __init__(self, backbone=False, bn=False, initializer='he_normal', regularizer=None, name="ERN", **kwargs):
        super().__init__(name=name, **kwargs)
        self.initializer = initializer
        self.regularizer = regularizer
        self.backbone = backbone
        self.bn = bn

        self.enc_edge_block6_upconv1 = self.conv2dtranspose(filters=256 if backbone else 496, name=f"enc_edge_block5_upconv1")
        self.enc_edge_block5_upconv1 = self.conv2dtranspose(filters=128 if backbone else 240, name=f"enc_edge_block4_upconv1")
        self.enc_edge_block4_upconv1 = self.conv2dtranspose(filters=64 if backbone else 112, name=f"enc_edge_block3_upconv1")
        self.enc_edge_block3_upconv1 = self.conv2dtranspose(filters=64 if backbone else 48, name=f"enc_edge_block2_upconv1")
        self.enc_edge_block2_upconv1 = self.conv2dtranspose(filters=3 if backbone else 16, name=f"enc_edge_block1_upconv1")
        self.enc_edge_conv1 = self.conv2d(filters=128, name=f"enc_edge_conv1")
        self.enc_edge_bn1 = layers.BatchNormalization(name=f"enc_edge_bn1")
        self.enc_edge_relu1 = layers.ReLU(name=f"enc_edge_relu1")
        self.enc_edge_conv2 = self.conv2d(filters=128, name=f"enc_edge_conv2")
        self.enc_edge_bn2 = layers.BatchNormalization(name=f"enc_edge_bn2")
        self.enc_edge_relu2 = layers.ReLU(name=f"enc_edge_relu2")
        self.enc_edge_conv3 = self.conv2d(filters=64, name=f"enc_edge_conv3")
        self.enc_edge_bn3 = layers.BatchNormalization(name=f"enc_edge_bn3")
        self.enc_edge_relu3 = layers.ReLU(name=f"enc_edge_relu3")
        self.enc_edge_conv4 = layers.Conv2D(filters=1, data_format="channels_last", kernel_size=(2, 2), activation="sigmoid", padding='same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, strides=(1, 1), name=f"enc_edge_conv4")

        self.dec_edge_block5_upconv1 = self.conv2dtranspose(filters=3 if backbone else 16, name=f"dec_edge_block5_upconv1")
        self.dec_edge_block4_upconv1 = self.conv2dtranspose(filters=64 if backbone else 48, name=f"dec_edge_block4_upconv1")
        self.dec_edge_block3_upconv1 = self.conv2dtranspose(filters=64 if backbone else 112, name=f"dec_edge_block3_upconv1")
        self.dec_edge_block2_upconv1 = self.conv2dtranspose(filters=128 if backbone else 240, name=f"dec_edge_block2_upconv1")
        self.dec_edge_block1_upconv1 = self.conv2dtranspose(filters=256 if backbone else 496, name=f"dec_edge_block1_upconv1")
        self.dec_edge_conv1 = self.conv2d(filters=128, name=f"dec_edge_conv1")
        self.dec_edge_bn1 = layers.BatchNormalization(name=f"dec_edge_bn1")
        self.dec_edge_relu1 = layers.ReLU(name=f"dec_edge_relu1")
        self.dec_edge_conv2 = self.conv2d(filters=128, name=f"dec_edge_conv2")
        self.dec_edge_bn2 = layers.BatchNormalization(name=f"dec_edge_bn2")
        self.dec_edge_relu2 = layers.ReLU(name=f"dec_edge_relu2")
        self.dec_edge_conv3 = self.conv2d(filters=64, name=f"dec_edge_conv3")
        self.dec_edge_bn3 = layers.BatchNormalization(name=f"dec_edge_bn3")
        self.dec_edge_relu3 = layers.ReLU(name=f"dec_edge_relu3")
        self.dec_edge_conv4 = layers.Conv2D(filters=1, data_format="channels_last", kernel_size=(2, 2), activation="sigmoid", padding='same', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, strides=(1, 1), name=f"dec_edge_conv4")

    def conv2d(self, filters=64, kernel_size=(3, 3), strides=(1, 1), name=None):
        return layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding='same', strides=strides, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, data_format="channels_last", name=name)

    def conv2dtranspose(self, filters=64, activation="relu", kernel_size=(2, 2), strides=(2, 2), name=None):
        return layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation=activation, strides=strides, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, data_format="channels_last", name=name)

    def call(self, encoder_outputs, decoder_outputs, training=False):
        x = self.enc_edge_block6_upconv1(encoder_outputs[-1])
        x = layers.Add()([encoder_outputs[-2], x])
        x = self.enc_edge_block5_upconv1(x)
        x = layers.Add()([encoder_outputs[-3], x])
        x = self.enc_edge_block4_upconv1(x)
        x = layers.Add()([encoder_outputs[-4], x])
        x = self.enc_edge_block3_upconv1(x)
        x = layers.Add()([encoder_outputs[-5], x])
        x = self.enc_edge_block2_upconv1(x)
        x = layers.Add()([encoder_outputs[-6], x])
        x = self.enc_edge_conv1(x)
        x = self.enc_edge_bn1(x) if self.bn else x
        x = self.enc_edge_relu1(x)
        x = self.enc_edge_conv2(x)
        x = self.enc_edge_bn2(x) if self.bn else x
        x = self.enc_edge_relu2(x)
        x = self.enc_edge_conv3(x)
        x = self.enc_edge_bn3(x) if self.bn else x
        x = self.enc_edge_relu3(x)
        enc_edge_output = self.enc_edge_conv4(x)

        y = self.dec_edge_block1_upconv1(decoder_outputs[0])
        y = layers.Add()([decoder_outputs[1], y])
        y = self.dec_edge_block2_upconv1(y)
        y = layers.Add()([decoder_outputs[2], y])
        y = self.dec_edge_block3_upconv1(y)
        y = layers.Add()([decoder_outputs[3], y])
        y = self.dec_edge_block4_upconv1(y)
        y = layers.Add()([decoder_outputs[4], y])
        y = self.dec_edge_block5_upconv1(y)
        y = layers.Add()([decoder_outputs[5], y])
        y = self.dec_edge_conv1(y)
        y = self.dec_edge_bn1(y) if self.bn else y
        y = self.dec_edge_relu1(y)
        y = self.dec_edge_conv2(y)
        y = self.dec_edge_bn2(y) if self.bn else y
        y = self.dec_edge_relu2(y)
        y = self.dec_edge_conv3(y)
        y = self.dec_edge_bn3(y) if self.bn else y
        y = self.dec_edge_relu3(y)
        dec_edge_output = self.dec_edge_conv4(y)

        return enc_edge_output, dec_edge_output

@tf.keras.utils.register_keras_serializable()
class MyUnetModel(keras.Model):
    def __init__(self, num_classes=7, size_image=512, backbone=False, training_up_to_layer=37, mini_blocks=3, ern=False, dropout=None, batch_norm=False, initializer='he_normal', regularizer=None, name="MyUnetModel", **kwargs):
        super().__init__(name=name, **kwargs)
        self.size_image = size_image
        self.ern = ern

        self.encoder = Encoder(size_image=size_image, backbone=backbone, training_up_to_layer=training_up_to_layer, mini_blocks=mini_blocks, dropout=dropout, bn=batch_norm, initializer=initializer, regularizer=regularizer)
        self.bottleneck = Bottleneck(backbone=backbone, dropout=dropout, bn=batch_norm, initializer=initializer, regularizer=regularizer)
        self.decoder = Decoder(backbone=backbone, mini_blocks=mini_blocks, dropout=dropout, bn=batch_norm, initializer=initializer, regularizer=regularizer)
        self.ern_net = ERN(backbone=backbone, bn=batch_norm, initializer=initializer, regularizer=regularizer)

    def call(self, inputs):
        encoder_output, skip_connections_encoder = self.encoder(inputs)
        bottleneck_output = self.bottleneck(encoder_output)
        decoder_output, skip_connections_decoder = self.decoder(bottleneck_output, skip_connections_encoder)
        if self.ern:
            enc_edge_output, dec_edge_output = self.ern_net(skip_connections_encoder, skip_connections_decoder)
            return {"segmentation": decoder_output, "edge_detection_encoder": enc_edge_output, "edge_detection_decoder": dec_edge_output}
        else:
            return decoder_output

    def build_graph(self):
        x = layers.Input(shape=(self.size_image, self.size_image, 3))
        return keras.Model(inputs=[x], outputs=self.call(x))

def dice_coefficient(y_true, y_pred):
    intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis=[3, 2, 1])
    dn = keras.backend.sum(keras.backend.square(y_true) + keras.backend.square(y_pred), axis=[3, 2, 1]) + keras.backend.epsilon()
    return keras.backend.mean(2 * intersection / dn)

def dice_loss(y_true, y_pred):
    intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis=[3, 2, 1])
    dn = keras.backend.sum(keras.backend.square(y_true) + keras.backend.square(y_pred), axis=[3, 2, 1]) + keras.backend.epsilon()
    dl = 2 * intersection / dn
    return - keras.backend.mean(dl)

class CyclicLR(keras.callbacks.Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            self.model.optimizer.learning_rate.assign(self.base_lr)
        else:
            keras.backend.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(self.model.optimizer.learning_rate.numpy())
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.model.optimizer.learning_rate.assign(self.clr())

if __name__ == '__main__':
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

    train_dataset = CustomDataset(train_df, DATA_DIR, augmentation=augmentation)
    valid_dataset = CustomDataset(valid_df, DATA_DIR, training=False)
    test_dataset = CustomDataset(test_df, DATA_DIR, training=False)

    unet_model = MyUnetModel(num_classes=6,
                         size_image=256,
                         backbone=True,
                         ern=True,
                         dropout=0.2,
                         batch_norm=True,
                         regularizer=keras.regularizers.L2(l2=1e-5)
                        )

    lr_init = 1e-6
    max_lr = 8e-5
    clr = CyclicLR(base_lr=lr_init, max_lr=max_lr, step_size=len(train_dataset)*2, mode='triangular')
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=5)
    checkpoint_filepath = 'new_model.h5'
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    unet_model.compile(loss={
                         "segmentation": dice_loss,
                         "edge_detection_encoder": sm.losses.BinaryFocalLoss(alpha=0.25, gamma=2.0),
                         "edge_detection_decoder": sm.losses.BinaryFocalLoss(alpha=0.25, gamma=2.0),
                        },
                   optimizer=keras.optimizers.Adam(learning_rate=lr_init),
                   metrics={
                       "segmentation": [dice_coefficient],
                       "edge_detection_encoder": [keras.metrics.BinaryCrossentropy()],
                       "edge_detection_decoder": [keras.metrics.BinaryCrossentropy()],
                       }
                  )

    history = unet_model.fit(train_dataset,
                         validation_data=valid_dataset,
                         epochs=100,
                         callbacks=[
                             early_stop,
                             ckpt_callback,
                             clr
                         ],
                         batch_size=16,
                         validation_batch_size=16,
                         verbose=1
                        )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['segmentation_dice_coefficient'])
    plt.plot(history.history['val_segmentation_dice_coefficient'])
    plt.title('Model Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()

    results = unet_model.evaluate(test_dataset)
    print(f'Test Loss: {results[0]}')
    print(f'Test Dice Coefficient: {results[1]}')

    def plot_predictions(model, dataset):
        n_images = 5
        
        images, masks = next(iter(dataset))
        preds = model.predict(images)
        
        plt.figure(figsize=(20, 10))
        
        for i in range(n_images):
            plt.subplot(3, n_images, i+1)
            plt.imshow(images[i])
            plt.title('Image')
            plt.axis('off')
            
            plt.subplot(3, n_images, i+1+n_images)
            plt.imshow(np.argmax(masks["segmentation"][i], axis=-1), cmap='jet')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(3, n_images, i+1+2*n_images)
            plt.imshow(np.argmax(preds["segmentation"][i], axis=-1), cmap='jet')
            plt.title('Prediction')
            plt.axis('off')
            
        plt.show()

    plot_predictions(unet_model, test_dataset)
