import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)
if img is None: raise FileNotFoundError("Put image.jpg in this folder")
img = cv2.resize(img, (200,200))

def apply_kernel(a, k):
    h,w = a.shape
    out = np.zeros((h,w), float)
    for i in range(1,h-1):
        for j in range(1,w-1):
            out[i,j] = np.sum(a[i-1:i+2,j-1:j+2]*k)
    return out

def normalize(a):
    a = np.abs(a)
    return (a*255/a.max()).astype(np.uint8) if a.max() else np.zeros_like(a,dtype=np.uint8)

def show(name,a):
    cv2.imshow(name,a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

smooth = apply_kernel(img.astype(float), np.array([[1,2,1],[2,4,2],[1,2,1]])/16)
sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
px = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
py = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
lap = np.array([[0,1,0],[1,-4,1],[0,1,0]])
sobel = normalize(np.hypot(apply_kernel(smooth,sx),apply_kernel(smooth,sy)))
prewitt = normalize(np.hypot(apply_kernel(smooth,px),apply_kernel(smooth,py)))
laplacian = normalize(apply_kernel(smooth,lap))
for name,a in [("Original",img),("Gaussian",normalize(smooth)),("Sobel",sobel),("Prewitt",prewitt),("Laplacian",laplacian)]:
    show(name,a)
