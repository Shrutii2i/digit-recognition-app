from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)
model = keras.models.load_model('digit_model.h5')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']

    # Remove the "data:image/png;base64," prefix that canvas.toDataURL() includes
    image_data = image_data.split(',')[1]

    # Decode base64 into actual image bytes
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # 'L' = grayscale

    # Resize to 28x28 to match MNIST format
    image = image.resize((28, 28))

    # Invert colors: canvas is black-on-white, MNIST is white-on-black
    image_array = np.array(image).astype('float32')
    image_array = 255 - image_array  # invert
    image_array = image_array / 255.0  # normalize

    image_array = image_array.reshape(1, 28, 28, 1)
    

    # Reshape to match model's expected input: (1, 28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)

    # Run prediction
    prediction = model.predict(image_array)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({"digit": predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)
