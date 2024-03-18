from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import nltk
from app.mcq_generation import MCQGenerator
from downloader import FilesDownloder

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

FilesDownloder()
nltk.download('punkt')

MQC_Generator = MCQGenerator()

@app.route("/")
@cross_origin()
def hello():
    return json.dumps('Hello World!')


@app.route("/generate", methods=["POST"])
@cross_origin()
def generate():
    #postman
    # text = request.form['text']

    requestJson = json.loads(request.data)
    text = requestJson['text']
    count = 10 if requestJson['count'] == '' else int(requestJson['count'])
    
    questions = MQC_Generator.generate_mcq_questions(text, count)
    result = list(map(lambda x: json.dumps(x.__dict__), questions))

    return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)