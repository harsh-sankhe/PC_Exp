import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)
if img is None: raise FileNotFoundError("Put image.jpg in this folder")

def show(name, a):
    cv2.imshow(name, a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Histogram equalization from scratch
hist = np.zeros(256)
for pixel in img.flat:
    hist[pixel] += 1
cdf = np.cumsum(hist) / img.size
mapping = np.round(cdf*255).astype(np.uint8)
equalized = mapping[img]

# Contrast stretching from scratch
low, high = int(img.min()), int(img.max())
if high == low:
    stretched = img.copy()
else:
    stretched = ((img.astype(float)-low)*255/(high-low)).astype(np.uint8)

show("Original", img)
show("Histogram Equalization", equalized)
show("Contrast Stretching", stretched)
# Optional histograms (visualization only)
import matplotlib.pyplot as plt
for name, a in [("Original",img),("Equalized",equalized),("Stretched",stretched)]:
    plt.hist(a.ravel(), bins=256, range=(0,256), alpha=0.4, label=name)
plt.legend()
plt.show()
