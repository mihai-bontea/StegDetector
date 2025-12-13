from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from steganography_detector import SteganographyDetector

def train_regular(self):
    detector = SteganographyDetector(use_residuals=False)
    detector.build_model()
    detector.train("dataset", epochs=10)

    detector.model.save("model/steg_detector.h5")

def train_residuals():
    detector = SteganographyDetector(use_residuals=True)
    detector.build_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        ),
        ModelCheckpoint(
            "model/steg_detector_residual_best.h5",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    detector.train(
        "dataset",
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )

    detector.model.save("model/steg_detector_residual_final.h5")

if __name__ == "__main__":
    train_residuals()
    print("Model trained and saved successfully!")