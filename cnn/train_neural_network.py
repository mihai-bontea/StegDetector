import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

class SteganographyDetector:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size
        self.model = None

    def load_images(self, data_dir):
        images = []
        labels = []
        for label, folder in enumerate(["cover", "stego"]):
            path = os.path.join(data_dir, folder)
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(os.path.join(path, file)).convert('RGB')
                    img = img.resize(self.image_size)
                    images.append(np.array(img) / 255.0)
                    labels.append(label)
        return np.array(images), np.array(labels)

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, data_dir, epochs=10, batch_size=32):
        X, y = self.load_images(data_dir)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model is None:
            self.build_model()

        datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10)
        train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)

        history = self.model.fit(
            train_gen,
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=1
        )
        return history

    def predict_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size)
        img = np.expand_dims(np.array(img) / 255.0, axis=0)

        pred = self.model.predict(img)[0][0]
        return bool(pred > 0.5), float(pred)

if __name__ == "__main__":
    detector = SteganographyDetector()
    detector.build_model()
    detector.train(r"dataset")  # requires cover/ and stego/ folders

    detector.model.save("steg_detector.h5")
    # detector.model.load_weights("steg_detector.h5")

    # result, confidence = detector.predict_image("test_image.png")
    # print(f"Steganography detected: {result} (confidence: {confidence:.2f})")

    # result, confidence = detector.predict_image("sample.png")
    # print("Stego Detected" if result else "Clean", confidence)




    # load the model
    #==========================================================
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# # Load the trained model
# model = load_model("steg_detector.h5")

# # Define image preprocessing function
# def preprocess_image(image_path, image_size=(128, 128)):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize(image_size)
#     img = np.array(img) / 255.0
#     return np.expand_dims(img, axis=0)



# 3. Run inference (predict if image has steganography)
# def predict_steganography(model, image_path):
#     img = preprocess_image(image_path)
#     pred = model.predict(img)[0][0]
#     is_stego = pred > 0.5
#     print(f"Steganography detected: {is_stego} (confidence: {pred:.2f})")
#     return is_stego, pred


# image_path = "test_image.png"
# predict_steganography(model, image_path)