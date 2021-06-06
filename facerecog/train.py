import os
from PIL import Image
import numpy as np
import cv2 as cv
import pickle

class train:
    def __init__(self, image_dir):
        #name of current file ->
        self.Base_path = os.path.abspath(__file__)
        self.Base_dir = os.path.dirname(self.Base_path)
        self.image_dir = os.path.join(self.Base_dir, image_dir)


        #Model for face detection
        self.cascade_face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv.face.LBPHFaceRecognizer_create()

        self.current_id = 0
        self.label_id = {} #contains index of lables 
        self.y_labels = [] #contains label indexes
        self.x_train = [] #contains image data as numpy array

        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg"):
                    #path of each picture
                    paths = os.path.join(root, file)
                    #name of folder of each picture
                    labels = os.path.basename(os.path.dirname(paths)).lower()
                    # print(paths,  labels)
                    
                    if labels in self.label_id:
                        pass
                    else:
                        self.label_id[labels] = self.current_id
                        self.current_id += 1
                        
                    id = self.label_id[labels]
                    # print(id) #printing index of each folder/picture
                    # y_labels.append(labels) #some number
                    # print(y_labels)
                    # x_train.append(paths)  #verify this image, turn into numpy array, turn into gray
                    # print(x_path)
                    pil_image = cv.imread(paths)
                    pil_image = cv.resize(pil_image, (300,500))
                    gray_image = cv.cvtColor(pil_image, cv.COLOR_BGR2GRAY) 
                    
                    image_array = np.array(gray_image, dtype='uint8')
                    # print(image_array)
                    faces = self.cascade_face.detectMultiScale(image_array, 1.2, 3)

                    for (x,y,w,h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        self.x_train.append(roi)
                        self.y_labels.append(id)
                    #   cv.rectangle(pil_image,(x,y),(x+w,y+h), (0,0,255), 2)
                    #   cv.imshow('efe',pil_image)
                #cv.waitKey(0)

        self.y_labels = np.array(self.y_labels)        

train_model = train('Pictures_people')

# print(y_labels)
# print(x_train)                

# with open("labels.pickle", 'wb') as f:
#     pickle.dump(train_model.label_id, f)

# train_model.recognizer.train(train_model.x_train, (train_model.y_labels))
# train_model.recognizer.save("trainer.yml")
