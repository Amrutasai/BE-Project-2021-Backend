import numpy as np
import cv2 as cv
from keras.models import load_model

model = load_model('twof.h5')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
colors = {'neutral':(255, 255, 255), 'angry':(0, 0, 255), 'fear':(0, 0, 0), 'happy':(0, 255, 255), 'sad':(255, 0, 0), 'surprised':(255, 245, 0)}
#imotions = {0:'angry', 1:'fear', 2:'happy', 3:'sad', 4:'surprised', 5:'neutral'}
#imotions = {0:'angry', 3:'happy', 4:'sad', 5:'neutral', 1:'neutral'}
imotions = {0:'angry', 2: 'happy', 3:'sad', 4:'surprised', 5:'neutral'}
def convert_dtype(x):
    x_float = x.astype('float32')
    return x_float

def normalize(x):
    x_n = (x - 0)/(255)
    return x_n


def reshape(x):
    x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    return x_r

class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        #cam = cv.VideoCapture(0)
        _, fr = self.video.read()

        #fr.set(cv.CV_CAP_PROP_FRAME_WIDTH, 800)
        #fr.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 600)sss

        while True:
            #img = cam.read()[1]
            #cv.Flip(, flipMode=-1)
            #img1 = cv.flip(img,1)
            gray = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            #gray_flip = cv.flip(gray,1)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = fr[y:y+h, x:x+w]
                roi_gray = cv.resize(roi_gray, (48, 48), interpolation = cv.INTER_AREA)
                roi_gray = convert_dtype(np.array([roi_gray]))
                roi_gray = normalize(roi_gray)
                roi_gray = reshape(roi_gray)
                pr = model.predict(roi_gray)[0]
                cv.rectangle(fr, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                maxindex = int(np.argmax(pr))
                cv.putText(fr, imotions[maxindex], (x+20, y-60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            _, jpeg = cv.imencode('.jpg', fr)
            return jpeg.tobytes()
