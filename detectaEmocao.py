import cv2
import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix


#ESTE CÓDIGO COMENTADO ABAIXO REFERE-SE AO TREINAMENTO DO MODELO, COMO NÃO HÁ A NECESSIDADE (NESTE MOMENTO)
# DE TREINAR TODAS AS VEZES QUE FOR PRECISO REALIZAR ALGUMA REQUISIÇÃO, FIZEMOS TODOS OS TESTES E TREINOS
# POSSÍVEIS E UTILIZAMOS O ARQUIVO COM O MODELO JÁ TREINADO



# data = pd.read_csv('Material/fer2013/fer2013.csv')
#
# # OS pixels gerado da base de dados, vem como string, porem precisa
# # ser convertida para o formato numpy.array
#
# pixels = data['pixels'].tolist()
#
# # Agora a formatação desses valores
# # cria um for para percorrer cada pixels
#
# largura, altura = 48, 48
# faces = []
# amostras = 0
# for pixel_sequence in pixels:
#     face = [int(pixel) for pixel in pixel_sequence.split(' ')]
#     face = np.asarray(face).reshape(largura, altura)
#     faces.append(face.astype('float32'))
#
# # transforma o faces em um numpy.array para o tensorFlow
# faces = np.asarray(faces)
#
# # Adiciona uma dimensão a mais, onde vai indicar que está sendo trabalhado com
# # imagens cinza, se fosse imagem colorida seria 3.
# faces = np.expand_dims(faces, -1)
#
#
# # Agora os dados estão no formato que o tensorFlow vai receber
# # (qtd de imagens, altura, largura, qtd de canais)
#
# # Função que vai normalizar os dados na escala 0 e 1. Até então tem valores
# # entre 0 e 255, mas para a rede neural processar mais rapido
# # é preciso colocar na escala 0 e 1.
#
# def normalizar(x):
#     x = x.astype('float32')
#     x = x / 255.0
#     return x
#
#
# faces = normalizar(faces)
#
# emocoes = pd.get_dummies(data['emotion']).values
#
# # Agora dividir o modelo em conjuntos de treinamento, validação e teste
# # a base de dados de treinamento será utilizada para treinar a rede neural
#
# X_train, X_test, y_train, y_test = train_test_split(faces, emocoes, test_size=0.1, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)
#
# np.save('Material/mod_xtest', X_test)
# np.save('Material/mod_ytest', y_test)
#
# # O objetivo dessa arquitetura é extrair caracteristicas importantes na imagem
# # Rede neural de múltiplas camadas
# # A cada camada adicionada, a rede vai extraindo caracteristicas,
# # assim para cada proxima camada, ela será baseada na anterior.
#
# # Uma camada convolucional realiza o aprendizado de múltiplos filtros,
# # onde cada filtro - ou kernel - extrai uma informação da imagem
#
# num_features = 64
# num_labels = 7
# batch_size = 64
# epochs = 50
# width, height = 48, 48
#
# model = Sequential()
#
# # primeira camada de convulução
# model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu',
#                  input_shape=(width, height, 1), data_format='channels_last',
#                  kernel_regularizer=l2(0.01)))
# model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# # segunda camada de convulução
# model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# # terceira camada de convulução
# model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# # quarta camada de convulução
# model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# # Depois que obtem o max pooling, aplicar o flattening
# # acontece que os resultados estão em matrizes, o flatten vai pegar essa matriz
# # e transformar em uma unica coluna de vetor
#
# model.add(Flatten())
#
# model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2 * 2 * num_features, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2 * num_features, activation='relu'))
# model.add(Dropout(0.5))
#
# # saida da rede, nesse caso será 7 saida, uma saida para cada emoção
# model.add(Dense(num_labels, activation='softmax'))
#
# # model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
#               metrics=['accuracy'])
#
# arquivo_modelo = 'Material/modelo_01_expressoes.h5'
# arquivo_modelo_json = 'Material/modelo_01_expressoes.json'
#
# lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
# early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
# checkpointer = ModelCheckpoint(arquivo_modelo, monitor='val_loss', verbose=1, save_best_only=True)
#
# model_json = model.to_json()
# with open(arquivo_modelo_json, 'w') as json_file:
#     json_file.write(model_json)
#
# history = model.fit(np.array(X_train), np.array(y_train),
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(np.array(X_test), np.array(y_test)),
#                     shuffle=True,
#                     callbacks=[lr_reducer, early_stopper, checkpointer])
#
# scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
#
# true_y = []
# pred_y = []
# x = np.load('Material/mod_xtest.npy')
# y = np.load('Material/mod_ytest.npy')
#
# json_file = open(arquivo_modelo_json, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
#
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights(arquivo_modelo)
#
# y_pred = loaded_model.predict(x)
#
# yp = y_pred.tolist()
# yt = y.tolist()
# count = 0
#
# for i in range(len(y)):
#     yy = max(yp[i])
#     yyt = max(yt[i])
#     pred_y.append(yp[i].index(yy))
#     true_y.append(yt[i].index(yyt))
#     if yp[i].index(yy) == yt[i].index(yyt):
#         count += 1
#
# acc = (count / len(y)) * 100
#
# np.save('truey_mod01', true_y)
# np.save('predy_mod01', pred_y)


json_file = open('Material/modelo_04_expressoes.json', 'r')
loaded_model = model_from_json(json_file.read())
json_file.close()
loaded_model.load_weights('Material/modelo_04_expressoes.h5')

y_true = np.load('truey_mod01.npy')
y_pred = np.load('predy_mod01.npy')

cm = confusion_matrix(y_true, y_pred)
expressoes = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']


def retornaEmocao(rosto):
    imagem = rosto
    original = imagem.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('Material/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    emocao = ''
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = roi_gray.astype('float') / 255.0
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = loaded_model.predict(cropped_img)[0]
        print(expressoes[int(np.argmax(prediction))])

        emocao = expressoes[int(np.argmax(prediction))]


    print(emocao)
    return emocao
