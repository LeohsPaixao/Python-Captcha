import cv2
import os
import glob


files = glob.glob('dest_folder/*')
for file in files:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Em Preto e Branco
    _, new_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

    # Encontear os contornos de cada letra
    contours, _ = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    i = 0
    for rectangle in letters_region:
        x, y, width, height = rectangle
        letter_image = image[y-2:y+height+2, x-2:x+width+2]
        i += 1
        file_name = os.path.basename(file).replace(".png", f"letra{i}.png")
        cv2.imwrite(f'letters/{file_name}', letter_image)
        cv2.rectangle(final_image, (x-2, y-2), (x+width+2, y+height+2), (0, 0, 255), 1)
    file_name = os.path.basename(file)
    cv2.imwrite(f"letters_id/{file_name}", final_image)