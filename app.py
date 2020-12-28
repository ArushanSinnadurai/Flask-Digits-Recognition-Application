from flask import Flask, request
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import model_from_json 

from image_processing import preprocess
from utils import data_uri_to_cv2_img, value_invert

# Setting up the Flask app
app = Flask(__name__)

# Allow Cross-Origin Resource Sharing
cors = CORS(app)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

@app.route('/')
def api_root():
    return app.send_static_file('interface.html')

@app.route('/post-data-url', methods=['POST'])
@cross_origin()
def api_predict_from_dataurl():

    # Read the image data from a base64 data URL
    imgstring = request.form.get('data')

    # Convert to OpenCV image
    img = preprocess(data_uri_to_cv2_img(imgstring))

    # Normalize values to 0-1, convert to white-on-black,
    # reshape for input layer
    data = value_invert(img/255).reshape(1, 28, 28, 1) 

    pred = loaded_model.predict_classes(data)[0]

    print("Prediction requested! Returned " + str(pred))

    # Return the prediction
    return str(pred) 

if __name__ == '__main__':
    app.run(debug=True)