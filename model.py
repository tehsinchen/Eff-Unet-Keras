import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.applications import efficientnet


def conv_kernel_initializer(shape, dtype=K.floatx()):
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def conv_block(x, filter, dropout_rate):
    x = layers.Conv2D(filter, 3, activation=None, padding='same', kernel_initializer=conv_kernel_initializer)(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter, 3, activation=None, padding='same', kernel_initializer=conv_kernel_initializer)(x)
    if dropout_rate != 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def up_conv_block(x, filter, cat_block, dropout_rate):
    x = layers.Conv2DTranspose(filter, 2, strides=2, padding='same', kernel_initializer=conv_kernel_initializer)(x)
    if cat_block is not None:
        x = layers.concatenate([cat_block, x], axis=3)
    x = conv_block(x, filter, dropout_rate)
    return x


def efficient_unet(input_shape, pretrained_weight):

    encoder = efficientnet.EfficientNetB2(
        include_top=False,
        weights=None,
        input_shape=input_shape,
    )

    block_dict = {}
    for i, layer in enumerate(encoder.layers):
        if 'add' in layer.name:
            block_name = layer.name[:6]
            block_dict[block_name] = i

    block7 = encoder.layers[block_dict['block7']].output
    block5 = encoder.layers[block_dict['block5']].output
    conv8 = up_conv_block(x=block7, filter=512, cat_block=block5, dropout_rate=0)

    block3 = encoder.layers[block_dict['block3']].output
    conv9 = up_conv_block(x=conv8, filter=256, cat_block=block3, dropout_rate=0)

    block2 = encoder.layers[block_dict['block2']].output
    conv10 = up_conv_block(x=conv9, filter=128, cat_block=block2, dropout_rate=0)

    block1 = encoder.layers[block_dict['block1']].output
    conv11 = up_conv_block(x=conv10, filter=64, cat_block=block1, dropout_rate=0)

    conv12 = up_conv_block(x=conv11, filter=32, cat_block=None, dropout_rate=0)

    conv13 = layers.Conv2D(1, 1, activation='sigmoid', kernel_initializer=conv_kernel_initializer)(conv12)

    input_ = encoder.input
    model = Model(input_, conv13)
    model.summary()

    if pretrained_weight:
        model.load_weights(pretrained_weight)

    return model


if __name__ == '__main__':
    efficient_unet((256, 256, 1), pretrained_weight=None)

