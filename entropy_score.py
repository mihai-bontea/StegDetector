import numpy as np
from PIL import Image
from scipy.stats import entropy

def shannon_entropy(data):
    values, counts = np.unique(data, return_counts=True)
    return entropy(counts, base=2)

def lsb_entropy_score(image_path):
    # Convert to grayscale
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)

    # Extract LSBs
    lsb_plane = img_data & 1

    # Calculate entropy of the LSB plane
    ent = shannon_entropy(lsb_plane.flatten())

    # Typical LSB entropy {~0.1, 0.5}; => closer to 1 suggests hidden data
    # Normalize score to [0,1]
    score = min(ent / 1.0, 1.0)
    return score

score = lsb_entropy_score("image.jpg")
print(f"Steganography Confidence Score (based on LSB entropy): {score:.3f}")