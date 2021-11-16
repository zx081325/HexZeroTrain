from flask import Flask, request
from wsgiref.simple_server import make_server

app = Flask(__name__)
white_batch_num, black_batch_num = 0, 0


@app.route('/white_batch', methods=["POST"])
def get_white_batch():
    global white_batch_num
    white_batch_num += 1
    print(white_batch_num, request)
    batch = request.files.get('file')
    file_name = "batch/white_batch_" + str(white_batch_num)
    if batch is not None:
        batch.save(file_name)
    with open("batch/white_batch_num.log", "w+") as f:
        f.write(str(white_batch_num))
    return "200"


@app.route('/black_batch', methods=["POST"])
def get_black_batch():
    global black_batch_num
    black_batch_num += 1
    print(black_batch_num, request)
    batch = request.files.get('file')
    file_name = "batch/black_batch_" + str(black_batch_num)
    if batch is not None:
        batch.save(file_name)
    with open("batch/black_batch_num.log", "w+") as f:
        f.write(str(white_batch_num))
    return "200"


if __name__ == "__main__":
    with open("batch/white_batch_num.log", "w+") as f:
        f.write("0")
    with open("batch/black_batch_num.log", "w+") as f:
        f.write("0")
    server = make_server('10.0.8.9', 83, app)
    server.serve_forever()
    app.run()
