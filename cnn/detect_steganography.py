import tensorflow as tf
from steganography_detector import SteganographyDetector

detector = SteganographyDetector(image_size=(128, 128))
detector.model = tf.keras.models.load_model("model/steg_detector.h5")

image_path = "1.jpg"
result, confidence = detector.predict_image(image_path)

print(f"Steganography detected: {result} (confidence: {confidence:.2f})")