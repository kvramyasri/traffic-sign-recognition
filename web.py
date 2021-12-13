#import required libraries 
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
from playsound import  playsound
from translate import Translator


#ceate a flsk app
app = Flask(__name__)

#Method to predict the test image from upload option
def model_predict(image):
    image = cv2.resize(image, (32,32))
    image = image.astype('float')
    image /= 255.0
    image = np.array(image).reshape(-1, 32, 32, 3)
    vgg16_model = tf.keras.models.load_model('modelvgg16.h5')
    pred_arr = vgg16_model.predict(image[[0], :]) 
    x=np.argmax(pred_arr,axis=1)
    signs_dictionary = {"0" : "Speed limit (20km/h)","1": "Speed limit (30km/h)","2": "Speed limit (50km/h)",
    "3": "Speed limit (60km/h)","4": "Speed limit (70km/h)","5": "Speed limit (80km/h)","6" : "End of speed limit (80km/h)",
    "7" :"Speed limit (100km/h)", "8": "Speed limit (120km/h)","9": "No passing","10" : "No passing veh over 3.5 tons",
    "11" : "Right-of-way at intersection","12":"Priority road","13":"Yield","14":"Stop","15":"No vehicles","16":"Veh > 3.5 tons prohibited",
    "17":"No entry","18":"General caution","19": "Dangerous curve left","20": "Dangerous curve right","21":"Double curve","22":"Bumpy road",
    "23":"Slippery road","24":"Road narrows on the right","25":"Road work","26":"Traffic signals","27":"Pedestrians","28":"Children crossing",
    "29": "Bicycles crossing","30":"Beware of ice/snow","31":"Wild animals crossing","32": "End speed + passing limits",
    "33":"Turn right ahead","34":"Turn left ahead","35":"Ahead only","36":"Go straight or right","37":"Go straight or left","38":"Keep right",
    "39":"Keep left","40":"Roundabout mandatory","41":"End of no passing", "42": "End no passing veh > 3.5 tons"
    }
    output = ""
    for sign in signs_dictionary.keys():
        if int(sign) == x:
            output = signs_dictionary[sign]
            return output    

#when the request method us get reder template
@app.route('/', methods=['GET'])
def index():
   output = ""
   return render_template('index.html')

#when the request method is post perform handle form
@app.route('/', methods=['GET', 'POST'])
def handle_form():
    output = ""
    translation = ""
    selected =""
    if request.method == 'POST':
        if request.form.get("classify"):
            input_file = request.files['file']
            input_file.save(secure_filename("test.jpg"))
            test_img=cv2.imread("test.jpg")
            output=model_predict(test_img)
            return render_template('index.html',result=output)
        elif request.form.get("play"):
            input_file = request.files['file']
            input_file.save(secure_filename("test.jpg"))
            test_img=cv2.imread("test.jpg")
            output=model_predict(test_img)
            language='en'
            myobj=gTTS(text=output,lang=language,slow=True)
            myobj.save("welcome1.mp3")
            playsound("welcome1.mp3")
            return render_template('index.html',result=output)
        elif request.form.get("translate"):
            selected=request.form.get('language')
            translator= Translator(to_lang=selected)
            input_file = request.files['file']
            input_file.save(secure_filename("test.jpg"))
            test_img=cv2.imread("test.jpg")
            output=model_predict(test_img)
            translation = translator.translate(output)
            return render_template('index.html',result=output,translation=translation)
        return render_template('index.html',result=output)
    if request.method == 'GET':
        print("Welcome to traffic recognition")

if __name__ == "__main__":
    app.debug = True
    app.run()