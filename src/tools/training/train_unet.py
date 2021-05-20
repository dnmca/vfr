
import numpy as np

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from src.models.unet import unet
from src.data.segmentation import CLASSES, IMAGE_SIZE, CHANNELS, get_data


def jaccard(ytrue, ypred, smooth=1e-5):

    intersection = np.sum(np.abs(ytrue * ypred), axis=-1)
    union = np.sum(np.abs(ytrue) + np.abs(ypred), axis=-1)
    jac = (intersection + smooth) / (union-intersection+smooth)
    return np.mean(jac)


def mean_squared_error(y_true, y_pred):
    channel_loss = np.sum(np.square(y_pred - y_true), axis=-1)
    total_loss = np.mean(channel_loss, axis=-1)
    return total_loss


def train():


    optimizer = RMSprop(lr=1e-3)

    model = unet(input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=None)

    train_gen, valid_gen = get_data('/home/andrii/Data/thesis', 1)

    model.fit_generator(
        generator=train_gen,
        validation_data=valid_gen,
        epochs=5,
        callbacks=[ModelCheckpoint(filepath='my_model', monitor='val_loss')]
    )


if __name__ == '__main__':
    train()