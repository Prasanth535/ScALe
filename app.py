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

def analyze_leaf(image_path, reference_length_mm, reference_pixels):
    """Core leaf analysis function"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Image not loaded"
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create green color mask (adjust these values as needed)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "No leaf detected"
            
        # Get largest contour
        leaf_contour = max(contours, key=cv2.contourArea)
        
        # Calculate pixel-to-mm ratio
        px_to_mm = reference_length_mm / reference_pixels
        
        # Calculate area
        area_px = cv2.contourArea(leaf_contour)
        area_mm2 = area_px * (px_to_mm ** 2)
        
        # Get rotated rectangle dimensions
        rect = cv2.minAreaRect(leaf_contour)
        (_, _), (width_px, height_px), _ = rect
        length_mm = max(width_px, height_px) * px_to_mm
        width_mm = min(width_px, height_px) * px_to_mm
        
        # Create result visualization
        result_img = img.copy()
        cv2.drawContours(result_img, [leaf_contour], -1, (0, 255, 0), 2)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)  # Changed from np.int0 to np.int32
        cv2.drawContours(result_img, [box], -1, (0, 0, 255), 2)
        
        # Save result image
        result_filename = 'result_' + os.path.basename(image_path)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        return {
            'area_mm2': round(area_mm2, 2),
            'area_cm2': round(area_mm2/100, 2),
            'length_mm': round(length_mm, 2),
            'length_cm': round(length_mm/10, 2),
            'width_mm': round(width_mm, 2),
            'width_cm': round(width_mm/10, 2),
            'result_image': result_filename,
            'px_to_mm': round(px_to_mm, 4)
        }, None
        
    except Exception as e:
        return None, str(e)

@app.route("/leafcalculator", methods=['GET', 'POST'])
def leafcalculator():
    if request.method == 'POST':
        file = request.files.get('leaf_image')
        try:
            ref_length = float(request.form.get('reference_length_cm', 15)) * 10  # Convert cm to mm
            ref_pixels = float(request.form.get('reference_pixel_length', 500))
        except:
            return render_template("leafcalculator.html", 
                                error="Invalid reference values. Please enter numbers.")
        
        if not file or file.filename == '':
            return render_template("leafcalculator.html", error="No file uploaded.")
            
        if not allowed_file(file.filename):
            return render_template("leafcalculator.html", error="File type not allowed. Please upload a PNG, JPG or JPEG image.")
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results, error = analyze_leaf(filepath, ref_length, ref_pixels)
        
        if error:
            return render_template("leafcalculator.html", error=error)
            
        return render_template("leafcalculator.html", 
                            results=results, 
                            original_image=filename,
                            reference_length_cm=ref_length/10,
                            reference_pixel_length=ref_pixels)
    
    return render_template("leafcalculator.html")

if __name__ == "__main__":
    app.run(debug=True)
