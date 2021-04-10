import cv2
import os
import glob
from PIL import Image


def treated_image(ori_folder, dest_folder='dest_folder'):
    files = glob.glob(f"{ori_folder}/*")
    for file in files:
        image = cv2.imread(file)

        # Transformar a imagem em escala de cinza
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, treated_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_TRUNC or cv2.THRESH_OTSU)
        file_name = os.path.basename(file)
        cv2.imwrite(f"{dest_folder}/{file_name}", treated_image)

    files = glob.glob(f"{dest_folder}/*")
    for file in files:
        image = Image.open(file)
        image = image.convert('P')
        image2 = Image.new('P', image.size, 255)

        for x in range(image.size[1]):  # Para cada Coluna da Imagem
            for y in range(image.size[0]):  # Para cada Linha da Imagem
                pixel_color = image.getpixel((y, x))
                if pixel_color < 115:
                    image2.putpixel((y, x), 0)
            file_name = os.path.basename(file)
            image2.save(f'{dest_folder}/{file_name}')


if __name__ == "__main__":
    treated_image('bdcaptcha')
