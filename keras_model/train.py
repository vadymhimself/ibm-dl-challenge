import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from model import lenet
from keras_img_iterator import SingleDirectoryIterator

tf.app.flags.DEFINE_string(
    'data_file',
    '../sample/train.csv',
    'The path to the csv file with image ids and labels')

tf.app.flags.DEFINE_integer(
    'batch_size',
    32,
    'Training batch size'
)

tf.app.flags.DEFINE_integer(
    'image_size',
    256,
    'input_image_size'
)

FLAGS = tf.app.flags.FLAGS


def main(args):
    data_file = FLAGS.data_file
    batch_size = FLAGS.batch_size
    image_size = FLAGS.image_size

    meta_data = pd.read_csv(data_file, header=0)
    filenames = meta_data['image_id'].apply(lambda id: id + '.png')
    classes = set(meta_data['label'])
    num_samples = meta_data.shape[0]  # first dimension

    # this is the augmentation configuration we will use for training
    gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = SingleDirectoryIterator(
        directory='../data/train_img/',
        filenames=filenames,
        labels=meta_data['label'],
        image_data_generator=gen,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        seed=1337)

    model = lenet(len(classes), image_size)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_samples // batch_size,
        epochs=50)

    # validation_data=validation_generator,
    # validation_steps=800 // batch_size)

    model.save_weights(
        'model.h5')  # always save your weights after training or during training


if __name__ == "__main__":
    tf.app.run()
