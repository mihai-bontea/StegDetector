import cv2
import numpy as np
from scipy.ndimage import convolve
from pathlib import Path


class HighPassResidualGenerator:
    """
    This class converts a directory of regular cover/stego images to a directory of residuals obtained
    from the given images. (The noise caused by steganography use is amplified, for better results when training a CNN.)
    """
    def __init__(self):
        self.kernel = np.array(
            [
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        )

    def load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img.astype(np.float32)

    def compute_residual(self, img):
        residual = convolve(img, self.kernel, mode="reflect")
        return residual

    def normalize_residual(self, residual, clip_value=3.0):
        residual = np.clip(residual, -clip_value, clip_value)
        residual = (residual + clip_value) / (2 * clip_value)
        residual = (residual * 255).astype(np.uint8)
        return residual

    def convert_images_to_residual(self, src_dir, dst_dir):
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for img_path in src_dir.iterdir():
            if img_path.suffix.lower() not in supported_exts:
                continue

            try:
                img = self.load_image(img_path)
                residual = self.compute_residual(img)
                residual_img = self.normalize_residual(residual)

                out_path = dst_dir / img_path.name
                cv2.imwrite(str(out_path), residual_img)

            except Exception as e:
                print(f"Failed processing {img_path.name}: {e}")


if __name__ == "__main__":
    generator = HighPassResidualGenerator()

    generator.convert_images_to_residual("cnn/dataset/cover", "cnn/dataset/cover_residual")
    print("Residual dataset generation complete for cover images.")

    generator.convert_images_to_residual("cnn/dataset/stego", "cnn/dataset/stego_residual")
    print("Residual dataset generation complete for stego images.")