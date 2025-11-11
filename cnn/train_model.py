from steganography_detector import SteganographyDetector

detector = SteganographyDetector(image_size=(128, 128))
detector.build_model()

detector.train("dataset", epochs=10)

detector.model.save("model/steg_detector.h5")
print("Model trained and saved successfully!")