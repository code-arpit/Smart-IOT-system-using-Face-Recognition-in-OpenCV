import cv2 as cv
import pickle

class face_recog:
    def __init__(self, video_source, haar_cascade, trainer, labels={}):        
        self.video = cv.VideoCapture(video_source)
        self.cascade_face = cv.CascadeClassifier(haar_cascade)
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.recognizer.read(trainer)

        self.labels = labels
        with open('labels.pickle', 'rb') as f:
            labels_train = pickle.load(f)
            self.labels = {v:k for k,v in labels_train.items()}


        while True:
            ret, self.frame = self.video.read()
            frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            detections = self.cascade_face.detectMultiScale(frame_gray, 1.3, 5, minSize=(250,250), maxSize=(250,250))
        
            for (x,y,w,h) in detections:
                #Draw frames over detections
                # roi_rect -> Region of interest -> detection
                roi_rect = frame_gray[y:y+h, x:x+w]
                id_, conf = self.recognizer.predict(roi_rect)
                
                id_text = self.labels[id_]
                print(conf)
                if not conf>70 and conf<90:
                    id_text = "Not Recognized"
                    
                self.faceframe(x,y,w,h, id_text)
                  
            cv.imshow('frame', self.frame)
            if cv.waitKey(1) & 0xFF==ord('q'):
                break
    
        self.video.release()
        cv.destroyAllWindows()

    def faceframe(self, x, y, w, h, id_text):
        line1 = cv.line(self.frame, (x,y), (x+35,y), (0,255,0), 3)
        line2 = cv.line(self.frame, (x,y), (x,y+35), (0,255,0), 3)
        line3 = cv.line(self.frame, (x+w-35,y), (x+w,y), (0,255,0), 3)
        line4 = cv.line(self.frame, (x+w,y), (x+w,y+35), (0,255,0), 3)
        line5 = cv.line(self.frame, (x,y+h-35), (x,y+h), (0,255,0), 3)
        line6 = cv.line(self.frame, (x,y+h), (x+35,y+h), (0,255,0), 3)
        line7 = cv.line(self.frame, (x+w-35,y+h), (x+w,y+h), (0,255,0), 3)
        line8 = cv.line(self.frame, (x+w,y+h-35), (x+w,y+h), (0,255,0), 3)
        
        cv.putText(self.frame, (id_text.upper()) , (x+40,y+5), cv.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255))
        



video_source = 0
haar_cascade = 'haarcascade_frontalface.xml'
trainer = 'trainer.yml'

face = face_recog(video_source, haar_cascade, trainer)        
