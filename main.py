from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

model_path = 'model/food_01_03_23.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

labels_path = 'model/labels.txt'
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image'].read()

    image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    image = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output[0])
    label = labels[predicted_index]
    confidence = float(output[0, predicted_index])

    response = {'label': label, 'confidence': confidence}
    return jsonify(response)

@app.route('/')
def home():
   return 'Call /predict with the image using post method to recieve a json output of {label, confidence}'

if __name__ == '__main__':
    app.run(port=8000)
