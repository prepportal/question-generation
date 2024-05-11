from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import nltk
from app.mcq_generation import MCQGenerator
from app.fill_in_the_blanks import FIBGenerator
from app.ml_models.trueorfalse_generation.trueorfalse_generation import TrueorFalseGenerator
from app.modules.find_title import find_title
from downloader import FilesDownloader

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

FilesDownloader()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('brown')

MCQ_Generator = MCQGenerator()
FIB_Generator = FIBGenerator()
TrueorFalse_Generator = TrueorFalseGenerator()

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
    topic = find_title(requestJson['text'])
    text = requestJson['text']
    count = requestJson.get('count', 3)
    q_type = requestJson.get('type', 'mcq')
    
    if q_type == 'mcq':
        questions = MCQ_Generator.generate_mcq_questions(text, count)
        questionjson = []
        for question in questions:
            questionjson.append({
                "question": question.questionText,
                "option1": question.distractors[0] if len(question.distractors) > 0 else None,
                "option2": question.distractors[1] if len(question.distractors) > 1 else None,
                "option3": question.distractors[2] if len(question.distractors) > 2 else None,
                "answer": question.answerText.capitalize()
            })
    elif q_type == 'fib':
        questionjson = FIB_Generator.generate_fill_in_the_blanks(text, count)
    elif q_type == 'truefalse':
        questionjson = TrueorFalse_Generator.generate(text, count)
    return jsonify({"questions":questionjson, "topic": topic})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)