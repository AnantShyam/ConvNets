import PIL
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import numpy as np
import os


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 4),
                     activation='relu', input_shape=(500, 500, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    return model


def process_image(image):
    length, height = image.size

    # image size: 500 pixels by 523 pixels
    rgb_image = image.convert('RGB')

    red, green, blue = [], [], []
    for i in range(length):
        r_row, g_row, b_row = [], [], []
        for j in range(height):
            r, g, b = image.getpixel((i, j))
            r_row.append(r)
            g_row.append(g)
            b_row.append(b)
        red.append(r_row)
        green.append(g_row)
        blue.append(b_row)

    # concatenate red, green, blue channels into a single tensor
    x_train = [red, green, blue]
    x_train = np.array(x_train)
    return x_train


def build_dataset(files, is_train=False):
    x_training = []
    y = []

    for file in files:
        if 'cat' in file:
            y.append(1)
        elif 'dog' in file:
            y.append(0)

        img = Image.open(f'{file}')
        img = img.resize((500, 500))
        x = process_image(img)
        x = x.reshape((500, 500, 3))
        x_training.append(x)

    y = np.array(y)
    x_training = np.stack(x_training, axis=0)

    if is_train:
        return x_training, y
    else:
        return x_training, None


if __name__ == "__main__":

    train_files, test_files = [], []
    for file in os.listdir('images/'):
        file_dir = os.path.join('images/', file)
        if 'test' not in file:
            train_files.append(file_dir)
        else:
            test_files.append(file_dir)

    m = build_model()

    x_train, y = build_dataset(train_files, True)

    m.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    m.fit(x_train, y, epochs=50)

    x_test, _ = build_dataset(test_files, False)
    # y_test = m.predict_classes(x_test)

    prediction = np.argmax(m.predict(x_test), axis=1)
    prediction = "cat" if prediction[0] == 1 else "dog"

    print(f"Prediction: {prediction}")

