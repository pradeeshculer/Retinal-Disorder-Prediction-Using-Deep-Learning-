from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__, template_folder='templates')

# Load your CNN model
model = load_model("E:/Documents/Docs/Pradeesh/Academic/VESIT/CNN_project/frontend/Lord_cnn.h5")

# Define the signs, symptoms, and cure for each category
category_info = {
    "CNV": {
        "signs": "Blurred or distorted vision, dark or empty areas in vision, sudden worsening of symptoms.",
        "symptoms": "Vision problems such as blurriness, distorted vision, or sudden vision loss.",
        "cure": "Intravitreal injections, Photodynamic therapy (PDT), Laser surgery."
    },
    "DME": {
        "signs": "Blurred vision, floaters, difficulty in reading, fluctuating vision.",
        "symptoms": "Blurred vision, floaters, distorted vision, difficulty in reading.",
        "cure": "Intravitreal injections, Laser treatment, Vitrectomy."
    },
    "DRUSEN": {
        "signs": "Small yellow deposits under the retina, blurred or dim vision, distorted vision.",
        "symptoms": "Blurred or dim vision, distorted vision, difficulty in reading.",
        "cure": "Regular eye exams, Anti-VEGF therapy, Photodynamic therapy."
    },
    "NORMAL": {
        "signs": "No abnormal signs detected in retinal images.",
        "symptoms": "Generally, no specific symptoms are present in normal retinal images.",
        "cure": "Regular eye check-ups for preventive care."
    }
}

# Define the route for the combined page
@app.route('/')
def combined_page():
    return render_template('combined.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        img_file = request.files['image']

        # Read the contents of the file
        img_bytes = img_file.read()

        # Load image from bytes
        img = image.load_img(BytesIO(img_bytes), target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        predicted_class = class_names[np.argmax(prediction)]

        # Get signs, symptoms, and cure for the predicted category
        signs = category_info[predicted_class]["signs"]
        symptoms = category_info[predicted_class]["symptoms"]
        cure = category_info[predicted_class]["cure"]

        # Encode image to base64 string
        img_str = base64.b64encode(img_bytes).decode('utf-8')

        return jsonify({'prediction': predicted_class, 'signs': signs, 'symptoms': symptoms, 'cure': cure, 'image': img_str})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
