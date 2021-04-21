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

# Converter os rotulos com one-hot encoding , pois somente os rotulos são texto
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Salvar o labelbinarizer em um arquivo com o pickle
with open('model_labels.dat', 'wb') as archive_pickle:
    pickle.dump(lb, archive_pickle) # Qual variavel e qual o arquivo


# Criação do IA e treinamento dela
model = Sequential()

# Criar as camadas da rede neural
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



