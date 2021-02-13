import cv2

imagePath = 'D:\\Study Materials\\Study\\My Python Workspace\\Intro-to-ML-and-DL-with-Python\\CV - Face Detection\\Images\\image3.jpg'
cascadeClassifierPath = 'D:\\Study Materials\\Study\\My Python Workspace\\Intro-to-ML-and-DL-with-Python\\CV - Face Detection\\haarcascade_frontalface_alt.xml'

cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

image = cv2.imread(imagePath)
if image is not None:
    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
else:
    print("Empty Frame!")
    exit(1)

detectedFaces = cascadeClassifier.detectMultiScale(grayImage)

for(x,y, width, height) in detectedFaces:
    cv2.rectangle(image, (x, y), (x+width, y+height), (0,255,0), 10)

cv2.imwrite('Output.jpg',image)
