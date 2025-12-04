import cv2
import numpy as np
import matplotlib.pyplot as plt


class RSAnalyzer:
    """
    Performs Regular/Singular (R/S) analysis on grayscale images
    (Group pixels and analyze smoothness changes caused by embedding)
    """

    def __init__(self, block_size: int = 16, backend: str = "python"):
        self.block_size = block_size
        self.backend = backend.lower()
        if self.backend not in ("python", "cpp"):
            raise ValueError("Backend must be 'python' or 'cpp'.")

    @staticmethod
    def _discrimination(group: np.ndarray) -> int:
        return np.sum(np.abs(np.diff(group)))

    @staticmethod
    def _flip_lsb(value: int) -> int:
        return value ^ 1

    @classmethod
    def _flip_group(cls, group: np.ndarray) -> np.ndarray:
        return np.array([cls._flip_lsb(x) for x in group])
    
    def _classify_groups(self, block: np.ndarray) -> tuple[int, int]:
        flat = block.flatten()
        R = S = 0
        for i in range(0, len(flat), 4):
            group = flat[i:i + 4]
            if len(group) < 4:
                continue
            f_o = self._discrimination(group)
            f_f = self._discrimination(self._flip_group(group))
            if f_f > f_o:
                R += 1
            elif f_f < f_o:
                S += 1
        return R, S

    def _analyze_python(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        map_h, map_w = h // self.block_size, w // self.block_size
        rs_map = np.zeros((map_h, map_w), dtype=np.float32)

        for i in range(map_h):
            for j in range(map_w):
                block = img[
                    i * self.block_size:(i + 1) * self.block_size,
                    j * self.block_size:(j + 1) * self.block_size,
                ]
                R, S = self._classify_groups(block)
                total = R + S if (R + S) > 0 else 1
                rs_map[i, j] = abs(R - S) / total
        return rs_map

    def _analyze_cpp(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("C++ backend not yet implemented.")
        # def _analyze_cpp(self, img):
        # import rs_cpp
        # return rs_cpp.rs_analyze(img, self.block_size)

    def analyze(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        if self.backend == "python":
            return self._analyze_python(img)
        elif self.backend == "cpp":
            return self._analyze_cpp(img)
    
    def stego_confidence(self, rs_map: np.ndarray) -> float:
        """
        Returns a confidence score in the [0, 1] range, indicating likelihood of stego
        """
        mean_score = np.mean(rs_map)
        ratio_score = np.mean(rs_map > 0.15)
        variance_score = np.var(rs_map)

        # Normalize components
        mean_n = np.clip((mean_score - 0.05) / 0.15, 0, 1)
        var_n  = np.clip(variance_score / 0.02, 0, 1)

        confidence = (
            0.5 * ratio_score +
            0.3 * mean_n +
            0.2 * var_n
        )

        return float(np.clip(confidence, 0, 1))

    @staticmethod
    def make_heatmap_figure(rs_map: np.ndarray):
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(rs_map, cmap="inferno", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("RS Difference (|R−S| / (R+S))")

        ax.set_title("RS Analysis Heatmap — Suspicious Regions Highlighted")
        ax.axis("off")

        return fig

    @staticmethod
    def make_old_style_overlay_figure(img: np.ndarray, rs_map: np.ndarray):
        rs_norm = rs_map / rs_map.max() if rs_map.max() != 0 else rs_map

        overlay = cv2.applyColorMap(
            (255 * (1 - rs_norm)).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        overlay = cv2.resize(overlay, (img.shape[1], img.shape[0]))

        combined = cv2.addWeighted(
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6,
            overlay, 0.4, 0
        )

        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(combined_rgb)
        ax.set_title("RS Suspicious Area Overlay")
        ax.axis("off")

        return fig
