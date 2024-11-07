from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import FileField, SubmitField
from io import BytesIO
from PIL import Image

import base64
import tensorflow as tf
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env
filePathString = 'model/best_model.keras'

# Load the trained model
model = tf.keras.models.load_model(filePathString)

# Load class names from JSON file for prediction labels
with open("data/class_names.json", "r") as f:
    model.class_names = json.load(f)

# Initialize Flask app with configurations
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') # Secret key for session handling
app.config['MAX_CONTENT_LENGTH'] = 8 * 1000 * 1000  # Limit upload size to 8 MB

# Define a form class for file upload with validation
class UploadFileForm(FlaskForm):
    file = FileField(label="File", 
                     validators=[FileRequired(),
                                 FileAllowed(["png", "jpg", "jpeg"], "This file is not a valid image!")])
    submit = SubmitField(label="Upload File")

# Home page route
@app.route("/")
@app.route("/home")
def home_page():
    return render_template("index.html", active_page="home_page")

# Classifier page route
@app.route("/classifier", methods=['GET', 'POST'])
def classification_page():
    form = UploadFileForm()

    if form.validate_on_submit():
        file = form.file.data  # Retrieve the uploaded file

        # Opens and resizes image to match model input size
        img = Image.open(file)
        img = img.resize((224, 224))

        # Converts the image to a format suitable for the model
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predicts the class and retrieves the result
        prediction = model.predict(img_array)
        predicted_class_index = tf.argmax(prediction[0]).numpy()
        predicted_class_name = model.class_names[predicted_class_index]
        prediction_accuracy = prediction[0][predicted_class_index]
        prediction_text = f"Predicted Class: {predicted_class_name} - Accuracy: {prediction_accuracy:.2%}"

        # Encodes the image in base64 for display in the template
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Render results on the classified page
        return render_template(
            "classified.html",
            prediction_text=prediction_text,
            active_page="classification_page",
            img_data=img_str
        )

    # Render classifier page with form and class names
    return render_template(
        "classifier.html",
        form=form,
        active_page="classification_page",
        class_names=model.class_names
    )

# Model information page route
@app.route("/model_information")
def information_page():
    return render_template("model_info.html", active_page="information_page")

# Run the Flask app in debug mode for development
if __name__ == "__main__":
    app.run(debug=True)
