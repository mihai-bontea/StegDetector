import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load grayscale image ---
img = cv2.imread("foid.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found.")

# --- RS Analysis function for a small block ---
def discrimination(group):
    return np.sum(np.abs(np.diff(group)))

def flip_lsb(value):
    return value ^ 1

def flip_group(group):
    return np.array([flip_lsb(x) for x in group])

def classify_groups(block):
    flat = block.flatten()
    R = S = 0
    for i in range(0, len(flat), 4):
        group = flat[i:i+4]
        if len(group) < 4:
            continue
        f_o = discrimination(group)
        f_f = discrimination(flip_group(group))
        if f_f > f_o:
            R += 1
        elif f_f < f_o:
            S += 1
    return R, S

# --- Compute local RS scores per block ---
block_size = 16
h, w = img.shape
map_h, map_w = h // block_size, w // block_size
rs_map = np.zeros((map_h, map_w))

for i in range(map_h):
    for j in range(map_w):
        block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        R, S = classify_groups(block)
        total = R + S if (R + S) > 0 else 1
        rs_map[i, j] = abs(R - S) / total

# --- Normalize and plot heatmap ---
plt.figure(figsize=(8, 6))
plt.imshow(rs_map, cmap='inferno', interpolation='nearest')
plt.colorbar(label='RS Difference (|R−S| / (R+S))')
plt.title("RS Analysis Heatmap — Suspicious Regions Highlighted")
plt.axis("off")
plt.show()

# overlay = cv2.applyColorMap((255 * (1 - rs_map / rs_map.max())).astype(np.uint8), cv2.COLORMAP_JET)
# overlay = cv2.resize(overlay, (w, h))
# combined = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, overlay, 0.4, 0)
# cv2.imshow("RS Suspicious Area Overlay", combined)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
