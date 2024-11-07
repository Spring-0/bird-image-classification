import tensorflow as tf
from tensorflow.keras import models, layers, datasets
from keras._tf_keras.keras.applications.resnet50 import ResNet50
from keras._tf_keras.keras.applications.efficientnet import EfficientNetB0
from keras.api.utils import image_dataset_from_directory
from keras._tf_keras.keras.metrics import Precision, Recall, BinaryAccuracy
import cv2
from matplotlib import pyplot as plt
import numpy as np


# TODO: Multiple birds in image classify as not bird
def train_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    data = image_dataset_from_directory(r"C:\Users\Jac\Desktop\workspace\bin-bird-classification\data",
                                        image_size=(224, 224),
                                        shuffle=True)

    data = data.map(lambda x, y: (x / 255, y))

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    model.fit(train, epochs=20, validation_data=val, callbacks=[callback])

    model.save("model-94.keras")

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        x, y = batch
        yhat = model.predict(x)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

    print(f"Precision: {pre.result()}, Recall: {re.result()}, Accuracy: {acc.result()}")

def test_model(img_path):
    model = models.load_model("model/model-94.keras")

    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    resized_img = tf.image.resize(img, (224, 224))

    yhat = model.predict(np.expand_dims(resized_img/255, 0))

    if yhat > 0.5:
        print("Not Bird")
    else:
        print("Bird")


test_model("apple.webp")

