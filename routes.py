from flask import Flask, request
from flask_cors import CORS
from main import cadastraPessoa, cadastraCamera, cadastraEmocao, retornaPessoaDB, retornaCameraDB, retornaIdentificacao
import detectaEmocao

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/person", methods=["POST"])
def chamaCadastroPessoa():
    body = request.get_json()

    try:
        cadastraPessoa(body["name"], body["imageFace"])
        return body, 200
    except:
        return {"message": "Algo inesperado aconteceu!"}, 400


@app.route("/api/person", methods=["GET"])
def retornaPessoa():
    try:
        return {"pessoas": retornaPessoaDB()}, 200
    except:
        return {"message": "Algo inesperado aconteceu!"}, 400


@app.route("/api/camera", methods=["POST"])
def chamaCadastroCamera():
    body = request.get_json()

    try:
        cadastraCamera(body["title"], body["ip"])
        return body, 200
    except:
        return {"message": "Algo inesperado aconteceu!"}, 400


@app.route("/api/camera", methods=["GET"])
def retornaCamera():
    try:
        return {"cameras": retornaCameraDB()}, 200
    except:
        return {"message": "Algo inesperado aconteceu!"}, 400


@app.route("/api/identification", methods=["POST"])
def chamaCadastroEmocao():
    body = request.get_json()

    try:
        cadastraEmocao(body["cameraId"], body["capturedAt"], body["image"])
        return body, 200
    except:
        return {"message": "Algo inesperado aconteceu!"}, 400


@app.route("/api/identification", methods=["GET"])
def retornaEmocao():
    dateIni = request.args['dateIni']
    dateFin = request.args['dateFin']
    try:
        cameraId = request.args['cameraId']
    except:
        cameraId = ""

    try:
        return {"Identificacoes": retornaIdentificacao(dateIni, dateFin, cameraId)}, 200
    except:
        return {"message": "Algo inesperado aconteceu!"}, 400


if __name__ == '__main__':
    app.debug = True
    app.run()

detectaEmocao
