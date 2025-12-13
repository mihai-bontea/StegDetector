import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image


class SteganographyDetector:
    def __init__(self, image_size=(128, 128), use_residuals=False):
        """
        use_residuals:
            False -> train/predict on raw RGB images
            True  -> train/predict on grayscale residual images
        """
        self.image_size = image_size
        self.use_residuals = use_residuals
        self.model = None

        self.channels = 1 if use_residuals else 3

    def _load_single_image(self, path):
        if self.use_residuals:
            img = Image.open(path).convert("L")
        else:
            img = Image.open(path).convert("RGB")

        img = img.resize(self.image_size)
        img = np.array(img, dtype=np.float32) / 255.0

        if self.channels == 1:
            img = np.expand_dims(img, axis=-1)

        return img

    def load_images(self, data_dir):
        images = []
        labels = []

        folders = (
            ["cover_residual", "stego_residual"]
            if self.use_residuals
            else ["cover", "stego"]
        )

        for label, folder in enumerate(folders):
            path = os.path.join(data_dir, folder)

            for file in os.listdir(path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(path, file)
                    img = self._load_single_image(img_path)

                    images.append(img)
                    labels.append(label)

        return np.array(images), np.array(labels)

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(
                32, (3, 3),
                activation="relu",
                input_shape=(*self.image_size, self.channels)
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        return model

    def train(self, data_dir, epochs=10, batch_size=32, callbacks=None):
        X, y = self.load_images(data_dir)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if self.model is None:
            self.build_model()

        datagen = (
            ImageDataGenerator()
            if self.use_residuals
            else ImageDataGenerator(horizontal_flip=True, rotation_range=10)
        )

        train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)

        history = self.model.fit(
            train_gen,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict_image(self, image_path):
        img = self._load_single_image(image_path)
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)[0][0]
        return float(pred)
