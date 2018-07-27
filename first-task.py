import os
import os.path
import cv2
import glob
import imutils
from sklearn.model_selection import train_test_split


CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
SINGLE_LETTERS_FOLDER = "extracted_letter_images"
PREDICTION_FOLDER = "sampled_captcha_images"

image_captchas = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

# We are sampling from the whole test some examples to feed the trained network to assess prediction accuracy, so that this examples are not used in training nor in validation, as are unseen at prediction time
captcha_image_files, prediction_samples = train_test_split(image_captchas, test_size=0.15, random_state=0)
if not os.path.exists(PREDICTION_FOLDER):
  os.makedirs(PREDICTION_FOLDER)

# move prediction samples in a separate folder
for (i, prediction_sample) in enumerate(prediction_samples):
  filename = os.path.basename(prediction_sample)
  p = os.path.join(PREDICTION_FOLDER, filename)
  os.rename(os.path.abspath(prediction_sample), p)

counts = {}
for (i, captcha_image_file) in enumerate(captcha_image_files):
    # grab the base filename as the label
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    image = cv2.imread(captcha_image_file)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.copyMakeBorder(grayscale_image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
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
    for letter_bounding_box, letter_text in zip(single_letter_image, captcha_correct_text):
        x, y, w, h = letter_bounding_box
        extracted_letter_image = grayscale_image[y - 2:y + h + 2, x - 2:x + w + 2]
        save_path = os.path.join(SINGLE_LETTERS_FOLDER, letter_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, extracted_letter_image)

        counts[letter_text] = count + 1
