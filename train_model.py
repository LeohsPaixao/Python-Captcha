import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


data = []

labels = []

folder_letters_base = "letters_base"

images = paths.list_images(folder_letters_base)

for archive in images:
    labels = archive.split(os.path.sep)[-2] # seperando os dados, os texto, tirando o contrabarra
    image = cv2.imread(archive)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Padronizar a imagem em 20x20
    image = resize_to_fit(image, 20, 20)

    # Adicionar uma nova dimensão para o Keras poder ler a imagem
    image = np.expand_dims(image, axis=2)

    # Adicionar as listas de dados e de rotulos
    labels.append(labels)
    data.append(image)

data = np.array(data, dtype="float") / 255
labels = np.array(labels)

# Separação em dados de treino(75%) e dados de teste (25$)
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Converter com one-hot encoding os rotulos, ja que são texto
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)