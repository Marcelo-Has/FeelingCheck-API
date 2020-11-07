from manipulaImage import decodeImage
import numpy as np
import face_recognition


def retornaAtributos(rostos):
    atributos = []
    for rosto in rostos:
        atributos.append(retornaAtributo(rosto))

    return atributos


def retornaRostos(foto):
    img1 = decodeImage(foto)

    rostos_location = face_recognition.face_locations(img1, number_of_times_to_upsample=1, model='hog')

    rostos = []
    for i in range(len(rostos_location)):
        topo, direita, baixo, esquerda = rostos_location[i]
        rosto = img1[topo - 30: baixo + 15, esquerda - 15: direita + 15]
        rostos.append(rosto)

    return rostos


def verificaAtributo(atributoBase, atributo):
    if np.any(atributo) and np.any(atributoBase):
        return face_recognition.compare_faces([atributoBase], atributo)[0]
    else:
        return False


def retornaAtributo(rosto):
    encodings = face_recognition.face_encodings(rosto)
    if encodings:
        atributo = encodings[0]
    else:
        atributo = []
    return atributo
