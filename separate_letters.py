import cv2
import os
import glob


files = glob.glob('dest_folder/*')
for file in files:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Em Preto e Branco
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

    # Encontear os contornos de cada letra
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters_region = []

    # Filtrar os contornos que são realmente de letras
    for contour in contours:
        (x, y, widght, height) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 115:
            letters_region.append((x, y, widght, height))

    # Tood captcha tiver 5 letras salva e não, passa!
    if len(letters_region) != 5:
        continue

    # desenhar os contornos e separar as letras em arquivos individuais
    final_image = cv2.merge([image] * 3)

