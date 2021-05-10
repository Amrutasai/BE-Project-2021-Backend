from flask import Flask, render_template, Response
from flask import request
from predict_image import get_emotion

# import cv2 as cv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test' , methods=["GET"])
def test():
    return {'emotion':'test'}

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