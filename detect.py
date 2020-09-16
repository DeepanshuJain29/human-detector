import cv2
import imutils

# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Reading the Image
# image = cv2.imread('no_bike_no_person/no_person__no_bike_017.bmp')
image = cv2.imread('persons/person_339.bmp')

# Resizing the Image
image = imutils.resize(image,
					width=min(400, image.shape[1]))

# Detecting all the regions in the
# Image that has a pedestrians inside it
(regions, _) = hog.detectMultiScale(image,
									winStride=(4, 4),
									padding=(4, 4),
									scale=1.05)
print(regions)

if len(regions) != 0:
	print("Yes")
	cv2.imshow("Image", image)
	cv2.waitKey(0)
else:
	print("No")
	cv2.imshow("Image", image)
	cv2.waitKey(0)


# Showing the output Image


cv2.destroyAllWindows()
