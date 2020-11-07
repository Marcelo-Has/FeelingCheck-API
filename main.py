import pyodbc
from detectaFace import retornaAtributo, verificaAtributo, retornaRostos
from manipulaImage import decodeImage, encodeImage
from detectaEmocao import retornaEmocao


def retornaConexão():
    server = "45.163.78.16,4989"
    database = "FeelingCheckDb"
    username = "root"
    password = "root"

    string_conexao = "Driver={ODBC Driver 17 for SQL Server};Server=" + server + ";Database=" + database + ";UID=" + username + ";PWD=" + password
    conexao = pyodbc.connect(string_conexao)
    return conexao.cursor()


def cadastraPessoa(nome, fotobase64):
    img = decodeImage(fotobase64)
    atributo = retornaAtributo(img)

    cursor.execute("INSERT INTO Person (Name, ImageFace) VALUES ('" + nome + "', '" + fotobase64 + "')")
    cursor.execute("SELECT * FROM Person WHERE Name = '" + nome + "'")
    personId = salvaId(cursor)

    for i in range(len(atributo)):
        cursor.execute("INSERT INTO IdentificationFace VALUES (" + str(atributo[i]) + ", " + str(i + 1) + ", " + str(
            personId) + ")")

    cursor.execute("commit")


def cadastraCamera(title, ip):
    cursor.execute("INSERT INTO Camera (Title, Ip) VALUES ('" + title + "', '" + ip + "')")
    cursor.execute("commit")


def cadastraEmocao(id, dataHora, foto):
    rostos = retornaRostos(foto)

    for rosto in rostos:
        cameraId = id
        image = encodeImage(rosto).decode("utf-8")
        personId = verificaPessoa(rosto)
        capturedAt = dataHora
        emotion = retornaEmocao(rosto)

        cursor.execute("INSERT INTO CaptureEmotion (PersonId, CameraId, CapturedAt, Emotion, Image) VALUES (" + str(
            personId) + ", " + str(cameraId) + ", '" + str(capturedAt) + "', '" + str(emotion) + "', '" + str(
            image) + "')")
        cursor.execute("commit")


def verificaPessoa(rosto):
    cursor.execute("SELECT * FROM Person")
    resultadoPerson = cursor.fetchall()
    qtdPessoa = len(resultadoPerson)

    for i in range(qtdPessoa):
        personId = resultadoPerson[i][0]
        cursor.execute("SELECT Value FROM IdentificationFace WHERE PersonId = " + str(personId) + " ORDER BY Ordem")
        resultadoId = cursor.fetchall()
        atributo = []
        for valor in resultadoId:
            atributo.append(float(valor[0]))

        if verificaAtributo(atributo, retornaAtributo(rosto)):
            return personId
            break

    return 1


def salvaId(cursor):
    row = cursor.fetchall()
    return row[0][0]


def retornaPessoaDB():
    cursor.execute("SELECT * FROM Person")
    row = cursor.fetchall()
    pessoas = []

    for linha in range(len(row)):
        pessoa = {}
        for coluna in range(len(row[linha])):
            if coluna == 1:
                pessoa["name"] = row[linha][coluna]
            elif coluna == 2:
                pessoa["imageFace"] = row[linha][coluna]

        pessoas.append(pessoa)

    return pessoas


def retornaCameraDB():
    cursor.execute("SELECT * FROM Camera")
    row = cursor.fetchall()
    cameras = []

    for linha in range(len(row)):
        camera = {}
        for coluna in range(len(row[linha])):
            if coluna == 0:
                camera["_Id"] = row[linha][coluna]
            elif coluna == 1:
                camera["title"] = row[linha][coluna]
            elif coluna == 2:
                camera["ip"] = row[linha][coluna]

        cameras.append(camera)

    return cameras


def retornaIdentificacao(dateIni, dateFin, cameraId):
    if cameraId != "":
        cameraId = "AND CAMERAID = " + str(cameraId)

    cursor.execute(
        "SELECT PER.NAME, PER.IMAGEFACE, CAP.EMOTION, CAP.CAPTUREDAT, CAP.Image as captureImage, CAM.Title AS CameraTitle FROM CAPTUREEMOTION CAP INNER JOIN PERSON PER ON PER._ID = CAP.PERSONID INNER JOIN CAMERA CAM ON CAM._ID = CAP.CAMERAID WHERE 	CAPTUREDAT BETWEEN '" + str(
            dateIni) + "' AND '" + str(dateFin) + "' " + str(cameraId))

    row = cursor.fetchall()
    identAll = []

    for linha in range(len(row)):
        ident = {}
        identPerson = {}

        for coluna in range(len(row[linha])):
            if coluna == 0:
                identPerson["name"] = row[linha][coluna]
            elif coluna == 1:
                identPerson["imageFace"] = row[linha][coluna]
                ident["person"] = identPerson
            elif coluna == 2:
                ident["emotion"] = row[linha][coluna]
            elif coluna == 3:
                ident["capturedAt"] = row[linha][coluna]
            elif coluna == 4:
                ident["captureImage"] = row[linha][coluna]
            elif coluna == 5:
                ident["cameraTitle"] = row[linha][coluna]

        identAll.append(ident)

    return identAll


cursor = retornaConexão()
