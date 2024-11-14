import tensorflow as tf
from keras import models, layers
from keras._tf_keras.keras.applications.resnet50 import ResNet50
from keras._tf_keras.keras.applications.efficientnet import EfficientNetB0
from keras.api.utils import image_dataset_from_directory
from keras._tf_keras.keras.metrics import Precision, Recall, BinaryAccuracy
from keras._tf_keras.keras.optimizers import Adam
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


# TODO: Fix: Multiple birds in image classify as not bird
def train_model():
    # Base model for Transfer Learning
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Custom model 
    model = models.Sequential()
    model.add(base_model) #  Add the base model
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))


    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Get Dataset object from "data" directory, automatically pre-processes the images.
    data = image_dataset_from_directory("data",
                                        image_size=(224, 224),
                                        batch_size=32,
                                        shuffle=True)

    data = data.map(lambda x, y: (x / 255, y))

    # Split the data into training, validation, and testing portions.
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    
    # Train the model. passing in training and validation data
    model.fit(train, epochs=20, validation_data=val, callbacks=[callback])

    # Additional accuracy data retrieval
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

    # Model saving logic
    if acc.result() > 0.80:
        model.save(f"model/model-{base_model.name}-{acc.result()}.keras")
        print("Model saved")
    else:
        print(f"Model accuracy too low: {acc.result()}")


#  Function used to manually test the model by loading the saved model
def test_model(img_path, model_name):
    model = models.load_model(os.path.join("model", model_name))

    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    resized_img = tf.image.resize(img, (224, 224))

    yhat = model.predict(np.expand_dims(resized_img/255, 0))

    if yhat > 0.5:
        print("Not Bird")
    else:
        print("Bird")

    
def get_model_info(model, file):
    import json
    with open(file, "w+") as f:
        json.dump(model.get_config(), f, indent=3)
    print(f"Hyperparams saved to hyperparams.json")
    print(f"Loss Function: {model.loss}")
    print(f"Optimizer: {model.optimizer}")

if __name__ == "__main__":
    # train_model()
    # train_model("flying-bird.jpg", "model-94.keras")
    # get_model_info(models.load_model("model/model-94.keras"))
    
    test_model("data/test/apple.webp", "model-efficientnetb0-0.8671875.keras")

