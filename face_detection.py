# load the library using the import keyword
# OpenCV must be properly installed for this to work. If not, then the module will not load with an
# error message.

import cv2
import sys

# Gets the name of the image file (filename) from sys.argv

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# This creates the cascade classifcation from file 

faceCascade = cv2.CascadeClassifier(cascPath)

# The image is read and converted to grayscale

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The face or faces in an image are detected
# This section requires the most adjustments to get accuracy on face being detected.


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(1,1),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Detected {0} faces!".format(len(faces)))

# This draws a green rectangle around the faces detected


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces Detected", image)
cv2.waitKey(0)
