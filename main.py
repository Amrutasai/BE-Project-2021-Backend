from flask import Flask, render_template, Response
from emotion_faces import VideoCamera
from flask import request
from classifier import classifyImage
from predict_image import get_emotion

# import cv2 as cv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(emotion_faces):
    while True:
        frame = emotion_faces.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed' , methods=["GET"])
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test' , methods=["GET"])
def test():
    return {'emotion':'test'}

@app.route('/classify', methods=['POST'])
def classify():
    print(request, request.files)
    if (request.files['image']): 
        file = request.files['image']

        result = classifyImage(file)
        print('Model classification: ' + result)        
        return result

@app.route('/getemotion', methods=['POST'])
def getemotion():
    print(request, request.files)
    if (request.files['image']): 
        file = request.files['image']
        file.save("Current.png")
        result = get_emotion(file)
        print("RESULT",result)
        print('Model classification: ' , result)        
        return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)