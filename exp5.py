import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)
if img is None: raise FileNotFoundError("Put image.jpg in this folder")
img = cv2.resize(img, (150,150))
a = (img > 127).astype(np.uint8)

def morph(a, mode):
    h,w = a.shape
    padded = np.pad(a,1)
    out = np.zeros_like(a)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3,j:j+3]
            out[i,j] = np.all(region) if mode == 'erode' else np.any(region)
    return out.astype(np.uint8)

def remove_small(a, minimum=30):
    h,w = a.shape
    seen = np.zeros_like(a,bool)
    out = np.zeros_like(a)
    for i in range(h):
        for j in range(w):
            if not a[i,j] or seen[i,j]: continue
            stack = [(i,j)]
            seen[i,j] = True
            group = []
            while stack:
                x,y = stack.pop()
                group.append((x,y))
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        nx,ny = x+dx,y+dy
                        if 0<=nx<h and 0<=ny<w and a[nx,ny] and not seen[nx,ny]:
                            seen[nx,ny] = True
                            stack.append((nx,ny))
            if len(group)>=minimum:
                for x,y in group: out[x,y]=1
    return out

def skeletonize(a):
    # Zhang-Suen thinning from scratch
    a = a.copy()
    h,w = a.shape
    changed = True
    while changed:
        changed = False
        for step in (0,1):
            delete = []
            for i in range(1,h-1):
                for j in range(1,w-1):
                    if a[i,j]==0: continue
                    p = [a[i-1,j],a[i-1,j+1],a[i,j+1],a[i+1,j+1],a[i+1,j],a[i+1,j-1],a[i,j-1],a[i-1,j-1]]
                    n = sum(p)
                    transitions = sum(p[k]==0 and p[(k+1)%8]==1 for k in range(8))
                    if not (2<=n<=6 and transitions==1): continue
                    if step==0:
                        ok = p[0]*p[2]*p[4]==0 and p[2]*p[4]*p[6]==0
                    else:
                        ok = p[0]*p[2]*p[6]==0 and p[0]*p[4]*p[6]==0
                    if ok: delete.append((i,j))
            for i,j in delete: a[i,j]=0
            if delete: changed=True
    return a

def show(name,a):
    cv2.imshow(name, a*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

e = morph(a,'erode')
d = morph(a,'dilate')
for name,result in [("Binary",a),("Erosion",e),("Dilation",d),("Boundary",a-e),("Remove Small Objects",remove_small(a)),("Skeleton",skeletonize(a))]:
    show(name,result)
