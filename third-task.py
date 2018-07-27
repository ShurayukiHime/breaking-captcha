from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os
import glob


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
PREDICTION_FOLDER = "sampled_captcha_images"
count = 0

# loading
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILENAME)

image_captchas = glob.glob(os.path.join(PREDICTION_FOLDER, "*"))
for image_captcha in image_captchas:
    filename = os.path.basename(image_captcha)
    captcha_correct_text = os.path.splitext(filename)[0]
    
    image = cv2.imread(image_captcha)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.copyMakeBorder(grayscale_image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # necessary because of opencv version
    contours = contours[0] if imutils.is_cv2() else contours[1]
    single_letter_image = []

    # letter extraction
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # check if two letter are in a single rectangle
        if w / h > 1.25:
            half_width = int(w / 2)
            single_letter_image.append((x, y, half_width, h))
            single_letter_image.append((x + half_width, y, half_width, h))
        else:
            single_letter_image.append((x, y, w, h))

    if len(single_letter_image) != 4:
        continue

    # Sort the letters w r t x coordinate, not to make mistakes when comparing the label
    single_letter_image = sorted(single_letter_image, key=lambda x: x[0])
    # Save each letter separately
    output = cv2.merge([grayscale_image] * 3)
    
    predictions = []
    for letter_bounding_box in single_letter_image:
        x, y, w, h = letter_bounding_box
        extracted_letter_image = grayscale_image[y - 2:y + h + 2, x - 2:x + w + 2]
        # resize to 20x20 because it's the dimension of the input
        extracted_letter_image = resize_to_fit(extracted_letter_image, 20, 20)
        # we also need a 4d tensor
        extracted_letter_image = np.expand_dims(extracted_letter_image, axis=2)
        extracted_letter_image = np.expand_dims(extracted_letter_image, axis=0)
        prediction = model.predict(extracted_letter_image)
        # remember we used a label binarizer...
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print result
    captcha_text = "".join(predictions)
    if captcha_correct_text != format(captcha_text):
        count = count + 1
        print(captcha_correct_text + format(captcha_text))
        #cv2.imshow("Wrong prediction", output)
        #cv2.waitKey()

print("Number of incorrect predictions: ", count)
