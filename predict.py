import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utility import image_resize
from utility import sort_contours

model = load_model(r'trained_model.h5')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to image file")
args = vars(ap.parse_args())
imagePath = args["image"]

image = cv2.imread(imagePath)
image = image_resize(image, width=600)
# Converting to grayscale and apply Gaussian filtering
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding the image
ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)
dilate = cv2.dilate(im_th, None, iterations=15)


# Finding contours in the image
ctrs, h = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ctrs, _ = sort_contours(ctrs)

# Get rectangles contains each contour
rects = []
for ctr in ctrs:
     if cv2.contourArea(ctr)>100:
        rects.append(cv2.boundingRect(ctr) )


recognised_digits = []
for rect in rects:
    # Draw the rectangles
    x,y,w,h = rect[0],rect[1],rect[2],rect[3]
    cv2.rectangle(image, (x,y), (x+ w, y+h), (0, 255, 0), 3) 

    roi = image[y:y+h, x:x+w]
    
    grey = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 155, 255, cv2.THRESH_BINARY_INV)
    
    # Resize the image
    resized_digit = cv2.resize(thresh, (28,28))
    prediction = model.predict(resized_digit.reshape(1, 28, 28, 1)) 
    image = cv2.putText(image, str(np.argmax(prediction)), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, 
                        color=(255,0,0), thickness=2)
    recognised_digits.append(np.argmax(prediction))

print("The processed image is stored as output.jpg")
print('Recognised Number - ',end = "")
for digit in recognised_digits:
    print(digit,end = "")

cv2.imwrite("output.jpg", image)
cv2.imshow("Resulting Image with recognised number", image)
cv2.waitKey(0)
cv2.destroyAllWindows()