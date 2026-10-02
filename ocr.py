import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def extract_text(image):
    return pytesseract.image_to_string(image)
