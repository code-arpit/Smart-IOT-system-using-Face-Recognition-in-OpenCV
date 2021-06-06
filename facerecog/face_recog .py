import cv2 as cv
import pickle


video = cv.VideoCapture(0)
cascade_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open('labels.pickle', 'rb') as f:
    labels_train = pickle.load(f)
    labels = {v:k for k,v in labels_train.items()}


while True:
    ret, frame = video.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detections = cascade_face.detectMultiScale(frame_gray, 1.3, 5)
 
    for (x,y,w,h) in detections:
        #recognizer
        roi = frame_gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)
        
        if conf>80:
            print(id_)
            print(labels[id_])

        
        frame = cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        rectangle = cv.rectangle(frame, (x,y+h),(x+w,y+h+25), (0,255,0), -1)
        cv.putText(rectangle,(labels[id_].upper()) , (x,y+h+20), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
        

    cv.imshow('frame', frame)
    if cv.waitKey(10) & 0xFF==ord('q'):
        break


video.release()
cv.destroyAllWindows()
