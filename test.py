import cv2

# opening camera
cam = cv2.VideoCapture(0)
# capturing Image
result, image = cam.read()

cv2.imwrite("img.jpg", image)