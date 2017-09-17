import pandas as pd
import tensorflow as tf
import numpy as np
from model import convnet

from keras.models import save_model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras_img_iterator import SingleDirectoryIterator
# from data import SingleDirectoryIterator

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer

tf.app.flags.DEFINE_string(
    'data_file',
    '../data/train_sample.csv',
    'The path to the csv file with image ids and labels')

tf.app.flags.DEFINE_integer(
    'batch_size',
    32,
    'Training batch size'
)

tf.app.flags.DEFINE_integer(
    'image_size',
    128,
    'input_image_size'
)

FLAGS = tf.app.flags.FLAGS


def main(args):
    data_file = FLAGS.data_file
    batch_size = FLAGS.batch_size
    image_size = FLAGS.image_size
    num_epochs = 10

    meta_data = pd.read_csv(data_file, header=0)
    filenames = meta_data['image_id'].apply(lambda id: id + '.png').values
    labels = meta_data['label'].values
    classes = list(set(labels))

    # split into test and validation
    files_train, files_validate, labels_train, labels_validate = \
        train_test_split(filenames, labels, test_size=0.2, random_state=42)

    num_train_samples = files_train.shape[0]
    num_val_samples = files_validate.shape[0]
    num_classes = len(classes)

    # this is the augmentation configuration we will use for training
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is a similar generator, for validation data
    # only rescaling
    test_gen = ImageDataGenerator(rescale=1. / 255)

    train_iterator = SingleDirectoryIterator(
        directory='../data/train_img/',
        filenames=files_train,
        labels=labels_train,
        classes=classes,
        image_data_generator=train_gen,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        seed=1337)

    validation_iterator = SingleDirectoryIterator(
        directory='../data/train_img/',
        filenames=files_validate,
        labels=labels_validate,
        classes=classes,
        image_data_generator=test_gen,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        seed=1337)

    # initialize and compile the model
    model = convnet(num_classes, image_size)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Train the model
    model.fit_generator(
        train_iterator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_iterator,
        validation_steps=num_val_samples // batch_size)

    # test model

    test_iterator = SingleDirectoryIterator(
        directory='../data/train_img/',
        filenames=files_validate,
        image_data_generator=test_gen,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        shuffle=False)

    test_steps = num_val_samples // batch_size

    predictions = model.predict_generator(
        generator=test_iterator,
        steps=test_steps)


    # binarize labels
    encoder = LabelBinarizer()
    y_true = encoder.fit_transform(labels_validate)
    y_true = y_true[:batch_size*test_steps, :]  # crop to match iterator

    int_labels = np.argmax(predictions, axis=1)
    y_predicted = encoder.fit_transform(int_labels)

    score = f1_score(y_true, y_predicted, average='weighted')

    print("model scored {} on validation set".format(score))

    # always save your weights after training or during training
    model_id = np.random.randint(1e4)
    model_file = 'model_{}_{}.h5'.format(model_id, score)
    save_model(model, model_file)

    print('Training complete. model was saved as ', model_file)




if __name__ == "__main__":
    tf.app.run()
