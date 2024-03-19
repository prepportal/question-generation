from flask import Flask, request, jsonify
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
nltk.download('stopwords')
nltk.download('wordnet')

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
    count = requestJson.get('count', 3)
    q_type = requestJson.get('type', 'mcq')
    
    if q_type == 'mcq':
        questions = MQC_Generator.generate_mcq_questions(text, count)
        questionjson = []
        for question in questions:
            questionjson.append({
                "question": question.questionText,
                "option1": question.distractors[0] if len(question.distractors) > 0 else None,
                "option2": question.distractors[1] if len(question.distractors) > 1 else None,
                "option3": question.distractors[2] if len(question.distractors) > 2 else None,
                "answer": question.answerText
            })
    elif q_type == 'fib':
        questionjson = MQC_Generator.generate_fill_in_the_blanks(text, count)
    return jsonify(questionjson)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)