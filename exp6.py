import cv2
import numpy as np

img = cv2.imread("image.jpg",0)
if img is None: raise FileNotFoundError("Put image.jpg in this folder")
img = cv2.resize(img,(200,200))

def convolution(img,k):
    h,w = img.shape
    out = np.zeros((h,w),float)
    for i in range(1,h-1):
        for j in range(1,w-1):
            out[i,j] = np.sum(img[i-1:i+2,j-1:j+2]*k)
    return out

# Edge based segmentation: Sobel from scratch
kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
g = np.hypot(convolution(img.astype(float),kx),convolution(img.astype(float),ky))
g = g*255/g.max() if g.max() else g
edges = (g>60).astype(np.uint8)*255

# Region growing from scratch, starting at center pixel
h,w = img.shape
seed = (h//2,w//2)
region = np.zeros((h,w),np.uint8)
visited = np.zeros((h,w),bool)
stack = [seed]
region_mean = float(img[seed])
count = 0
while stack:
    x,y = stack.pop()
    if not (0<=x<h and 0<=y<w) or visited[x,y]: continue
    visited[x,y] = True
    if abs(float(img[x,y])-region_mean)>15: continue
    region[x,y] = 255
    count += 1
    region_mean += (float(img[x,y])-region_mean)/count
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        stack.append((x+dx,y+dy))

for name,a in [("Original",img),("Edge Segmentation",edges),("Region Growing",region)]:
    cv2.imshow(name,a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
