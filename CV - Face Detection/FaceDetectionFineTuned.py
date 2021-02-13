import cv2

imagePath = 'D:\\Study Materials\\Study\\My Python Workspace\\Intro-to-ML-and-DL-with-Python\\CV - Face Detection\\Images\\image3.jpg'
cascadeClassifierPath = 'D:\\Study Materials\\Study\\My Python Workspace\\Intro-to-ML-and-DL-with-Python\\CV - Face Detection\\haarcascade_frontalface_alt.xml'

cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

img = cv2.imread(imagePath)
if img is not None:
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
else:
    print("Empty Frame!")
    exit(1)

detectedFaces = cascadeClassifier.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=4, minSize=(15,15))

for(x,y,width, height) in detectedFaces:
    cv2.rectangle(img, (x,y), (x+width, y+height), (0,255,0), 10)

cv2.imwrite('Output2.jpg',img)
