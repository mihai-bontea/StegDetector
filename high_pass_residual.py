import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, uniform_filter


class HighPassResidualSteganalysis:
    def __init__(self, window_size=8):
        """
        window_size: size of the sliding window for local variance
        """
        self.window_size = window_size

        # Common steganalysis high-pass filter (Laplacian-like)
        self.kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    def load_image(self, path):
        """
        Loads the image as grayscale float32 [0..255]
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img.astype(np.float32)

    def compute_residual(self, img):
        """
        Apply high-pass filter to obtain noise residual
        """
        residual = convolve(img, self.kernel, mode='reflect')
        return residual

    def local_variance(self, residual):
        """
        Compute local variance using the sliding window
        var = E[x^2] - (E[x])^2
        """
        w = self.window_size

        mean = uniform_filter(residual, size=w)
        mean_sq = uniform_filter(residual**2, size=w)

        variance = mean_sq - mean**2
        return variance

    def normalize(self, arr):
        """
        Normalize to 0..1 for visualization
        """
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr
    
    def stego_confidence(self, var_map):
        """
        Returns stego likelihood in [0,1]
        """
        mean_v = np.mean(var_map)
        std_v  = np.std(var_map)
        tail_ratio = np.mean(var_map > 2 * mean_v)

        # Normalization (empirical but stable)
        mean_n = np.clip((mean_v - 5) / 15, 0, 1)
        std_n  = np.clip(std_v / 20, 0, 1)

        confidence = 0.5 * tail_ratio + 0.3 * mean_n + 0.2 * std_n
        return float(np.clip(confidence, 0, 1))

    def analyze(self, img_path):
        """
        Full pipeline:
        - load
        - residual
        - variance map
        - normalized heatmap
        """
        img = self.load_image(img_path)
        residual = self.compute_residual(img)
        var_map = self.local_variance(residual)
        confidence_score = self.stego_confidence(var_map)
        heatmap = self.normalize(var_map)
        return confidence_score, residual, var_map, heatmap
    
    def make_figure(self, img, residual, var_map, heatmap, cmap="inferno"):
        fig = plt.figure(figsize=(14, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("Original Image")
        ax1.imshow(img, cmap="gray")
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("High-Pass Noise Residual")
        ax2.imshow(residual, cmap="gray")
        ax2.axis("off")

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("Local Residual Variance")
        ax3.imshow(var_map, cmap="gray")
        ax3.axis("off")

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("Suspicion Heatmap Overlay")
        ax4.imshow(img, cmap="gray")
        ax4.imshow(heatmap, cmap=cmap, alpha=0.6)
        ax4.axis("off")

        fig.tight_layout()
        return fig