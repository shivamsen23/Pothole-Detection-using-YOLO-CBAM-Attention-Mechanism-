import numpy as np

def predict_image(model, image):
    img_array = np.array(image)

    results = model.predict(img_array, conf=0.25)

    # Get plotted image
    result_img = results[0].plot()

    # Count potholes
    count = len(results[0].boxes)

    return result_img, count