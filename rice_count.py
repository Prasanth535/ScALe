import cv2
import numpy as np

def count_rice_seeds(image_path):
    # Read and preprocess image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Area filtering (values from the PDF)
    lowest_area = 2891.0
    highest_area = 9258.0
    filtered_contours = [c for c in contours if lowest_area <= cv2.contourArea(c) <= highest_area]
    
    # Draw contours on original image (optional visualization)
    output_image = image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)
    
    # Print detailed information (like in the PDF)
    for i, contour in enumerate(filtered_contours):
        area = cv2.contourArea(contour)
        print(f'Contour {i}: Area = {area}')
    
    seed_count = len(filtered_contours)
    if seed_count > 0:
        print(f'Total number of seeds detected: {seed_count} (Rice)')
    else:
        print(f'Total number of seeds detected: {seed_count}')
    
    return seed_count

