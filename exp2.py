import cv2
import numpy as np
import math

img = cv2.imread("image.jpg", 0)
if img is None: raise FileNotFoundError("Put image.jpg in this folder")
img = cv2.resize(img, (150, 150))

def show(name, a):
    print(name)
    cv2.imshow(name, a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show("Original", img)
h, w = img.shape
mean = img.copy()
gaussian = img.copy()
median = img.copy()
bilateral = img.copy()
k = np.array([[1,2,1],[2,4,2],[1,2,1]])
for i in range(1, h-1):
    for j in range(1, w-1):
        region = img[i-1:i+2, j-1:j+2].astype(float)
        mean[i,j] = np.mean(region)
        gaussian[i,j] = np.sum(region*k)/16
        median[i,j] = np.median(region)
        total = 0
        weights = 0
        for x in range(-1,2):
            for y in range(-1,2):
                value = float(img[i+x,j+y])
                spatial = math.exp(-(x*x+y*y)/(2*2**2))
                intensity = math.exp(-((value-float(img[i,j]))**2)/(2*30**2))
                weight = spatial*intensity
                total += value*weight
                weights += weight
        bilateral[i,j] = total/weights
for name, result in [("Mean",mean),("Gaussian",gaussian),("Median",median),("Bilateral",bilateral)]:
    show(name, result)
