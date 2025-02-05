!pip install flask flask-cors pyngrok
!pip install flask-cors
!pip install pyngrok


from PIL import Image
import numpy as np

import tensorflow as tf



from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Make sure to import CORS
from pyngrok import ngrok 

import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for external access
model = tf.keras.models.load_model('mnist_model.h5')  # Load your saved model

def preprocess_image(image):
    # Process image for MNIST model (28x28 grayscale)
    img = Image.open(image).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # Invert colors (MNIST uses white-on-black)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Add batch dimension
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_array = preprocess_image(file)
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))

    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence
    })



# Start ngrok tunnel
public_url = ngrok.connect(5000).public_url
print(f"Public URL: {public_url}")  # Copy this URL

# Run Flask on 0.0.0.0 to allow external access
app.run(host='0.0.0.0', port=5000)
