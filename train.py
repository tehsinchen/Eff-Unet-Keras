import os
import math
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
#from model import model_unet
from model_effunet import efficient_unet


def step_decay_schedule(initial_lr, decay_factor, step_size):

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


def jaccard(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def get_data(folder_path):
    data_path = os.path.join(folder_path, 'x_train.npy')
    label_path = os.path.join(folder_path, 'y_train.npy')
    print('data loading...')
    with open(data_path, 'rb') as f:
        train_data = np.load(f)
    with open(label_path, 'rb') as f:
        label_data = np.load(f).astype(np.float32)
    return train_data, label_data


def get_split_data(train_data, label_data, batch_size):
    train_step = math.ceil(int(label_data.shape[0] * 0.8) / batch_size)
    train_size = train_step * batch_size
    x_train = train_data[:train_size, :, :, :]
    y_train = label_data[:train_size, :]

    val_x = train_data[train_size:, :, :, :]
    val_y = label_data[train_size:, :]
    val_step = math.ceil(val_y.shape[0] / batch_size)
    val_size = val_step * batch_size
    if val_size > val_y.shape[0]:
        val_step -= 1
        val_size = val_step * batch_size
    val_x = val_x[:val_size, :, :, :]
    val_y = val_y[:val_size, :]
    del train_data, label_data
    return x_train, y_train, val_x, val_y, train_step, val_step


def main():
    # find the current path of this program
    folder_path = os.getcwd().replace('code', 'data2')
    base_model_weight = os.path.join(folder_path, 'model_base.hdf5')
    model_path = os.path.join(folder_path, 'model', 'model_cell_mask-{epoch:02d}-{val_mean_io_u:.2f}.hdf5')

    # a strategy to use multiple GPU
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    devices = tf.config.experimental.list_physical_devices("GPU")
    print(f'found GPU devices: {devices}')

    # data loading and split into training & validation
    train_data, label_data = get_data(folder_path)
    batch_size = 25 * strategy.num_replicas_in_sync
    x_train, y_train, val_x, val_y, train_step, val_step = get_split_data(train_data, label_data, batch_size)
    print(x_train.shape, y_train.shape, val_x.shape, val_y.shape, train_step, val_step)

    # load the model in each GPU
    with strategy.scope():
        model = efficient_unet((256, 256, 1), base_model_weight)
        model.compile(optimizer=Adam(epsilon=1e-8),
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

    lr_sched = step_decay_schedule(initial_lr=0.001, decay_factor=0.9, step_size=150)
    model_checkpoint = ModelCheckpoint(model_path,
                                       monitor='val_mean_io_u',
                                       verbose=1,
                                       #save_weights_only=True,
                                       save_best_only=True,
                                       mode='max')
    model.fit(x_train,
              y_train,
              steps_per_epoch=train_step,
              epochs=7000,
              verbose=1,
              callbacks=[model_checkpoint, lr_sched],
              validation_data=(val_x, val_y),
              validation_steps=val_step,
              shuffle=True)


if __name__ == '__main__':
    main()
