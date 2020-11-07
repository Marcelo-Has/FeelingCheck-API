import base64
import cv2


def decodeImage(string):
    imageBytes = bytes(string, encoding="utf-8")
    i = open("img/temp.jpg", "wb")
    i.write(base64.b64decode(imageBytes))
    i.close()

    img = cv2.cvtColor(cv2.imread("img/temp.jpg"), cv2.COLOR_BGR2RGB)

    return img


def encodeImage(imagem):
    path = "img/rostotemp.jpg"
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, imagem)
    i = open(path, "rb")
    return base64.b64encode(i.read())
