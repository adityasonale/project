from keras.models import load_model
from time import sleep
import numpy as np
import cv2

from keras_preprocessing.image import img_to_array
from keras.preprocessing import image

faceClassifier = cv2.CascadeClassifier('D:\opencv-4.x\opencv-4.x\data\haarcascades_cuda\haarcascade_frontalface_default.xml')
classifier = load_model('D:\\vs code\.vscode\python\DeepLearning\models\emotionDetection.h5')

# load model here

emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']   # try changing the sequence

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+h]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        else:
            cv2.putText(frame,"no face",(30,80),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    cv2.imshow("Emotion Detector",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
