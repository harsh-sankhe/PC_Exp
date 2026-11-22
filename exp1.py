import cv2

img = cv2.imread("image.jpg")
if img is None:
    raise FileNotFoundError("Put image.jpg in this folder")

def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show("Original", img)
print("Height, Width, Channels:", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show("Grayscale", gray)
show("HSV", cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
show("Resized", cv2.resize(img, (300, 300)))
h, w = img.shape[:2]
crop = img[h//4:3*h//4, w//4:3*w//4]
show("Cropped", crop)
show("Contrast Enhanced", cv2.convertScaleAbs(img, alpha=1.8))
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show("Otsu Threshold", binary)
show("Canny Edges", cv2.Canny(gray, 100, 200))
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
marked = img.copy()
cv2.drawContours(marked, contours, -1, (0, 255, 0), 2)
show("Contours", marked)
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
show("Rotated", cv2.warpAffine(img, M, (w, h)))
cv2.imwrite("cropped_image.jpg", crop)
