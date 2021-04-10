import cv2
from Pillow import Image

methods = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV
]

image = cv2.imread("bdcaptcha/telanova0.png")

# Transformar a imagem em escala de cinza
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


i = 0
for method in methods:
    i += 1
    _, treated_image = cv2.threshold(
        image_gray, 127, 255, method or cv2.THRESH_OTSU)
    cv2.imwrite(f"method_test/treated_image_{i}.png", treated_image)
