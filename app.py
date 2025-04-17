import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from rice_count import count_rice_seeds
from maize_count import count_maize_seeds
from millet_count import count_millet_seeds
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/seedcounter", methods=['GET', 'POST'])
def seedcounter():
    seed_count = None
    seed_type = None
    image_path = None

    if request.method == 'POST':
        seed_type = request.form.get('seed_type')
        file = request.files.get('file')

        if not file or file.filename == '':
            flash("No file uploaded.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("File type not allowed. Please upload a PNG, JPG or JPEG image.", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_path = file_path

        try:
            if seed_type == "rice":
                seed_count = count_rice_seeds(file_path)
            elif seed_type == "maize":
                seed_count = count_maize_seeds(file_path)
            elif seed_type == "millet":
                seed_count = count_millet_seeds(file_path)
            else:
                flash("Invalid seed type selected.", "error")
                return redirect(request.url)
        except Exception as e:
            flash(f"Error processing image: {str(e)}", "error")
            return redirect(request.url)

    return render_template("seedcounter.html", seed_count=seed_count, seed_type=seed_type, image_path=image_path)

@app.route("/leafcalculator", methods=['GET', 'POST'])
def leafcalculator():
    area = None
    length = None
    width = None
    image_path = None
    error = None

    if request.method == 'POST':
        file = request.files.get('leaf_image')
        reference_length_cm = float(request.form.get('reference_length_cm', 0))
        reference_pixel_length = float(request.form.get('reference_pixel_length', 0))

        if not file or file.filename == '':
            error = "No file uploaded."
        elif not allowed_file(file.filename):
            error = "File type not allowed. Please upload a PNG, JPG or JPEG image."
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_path = file_path

            try:
                area, length, width = calculate_leaf_properties(file_path, reference_length_cm, reference_pixel_length)
                if area is None:
                    error = "Leaf detection failed. Try another image."
            except Exception as e:
                error = f"Error processing image: {str(e)}"

    return render_template("leafcalculator.html", area=area, length=length, width=width, image_path=image_path, error=error)

def calculate_leaf_properties(image_path, reference_length_cm, reference_pixel_length):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    leaf_contour = max(contours, key=cv2.contourArea)
    leaf_area_pixels = cv2.contourArea(leaf_contour)
    pixel_to_cm_ratio = (reference_length_cm / reference_pixel_length) ** 2
    leaf_area_cm2 = leaf_area_pixels * pixel_to_cm_ratio
    
    x, y, width_pixels, height_pixels = cv2.boundingRect(leaf_contour)
    leaf_length_cm = height_pixels * (reference_length_cm / reference_pixel_length)
    leaf_width_cm = width_pixels * (reference_length_cm / reference_pixel_length)
    
    return leaf_area_cm2, leaf_length_cm, leaf_width_cm

if __name__ == "__main__":
    app.run(debug=True)