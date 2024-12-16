import cv2
from PIL import Image


image_cat_path = 'Cat2.png'
image_cat = cv2.imread(image_cat_path)

cat_face_handler = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

cat_face_coordinates = cat_face_handler.detectMultiScale(image_cat)

for (x, y, w, h) in cat_face_coordinates:
    cv2.rectangle(image_cat, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('Cat 2', image_cat)

cat = Image.open(image_cat_path)

cv2.waitKey()
