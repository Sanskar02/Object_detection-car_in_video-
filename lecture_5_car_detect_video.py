import cv2

body_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

cap = cv2.VideoCapture('cars.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255), 2)
        cv2.imshow('Pedestrain', frame)
        
    if cv2.waitKey(1) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()
