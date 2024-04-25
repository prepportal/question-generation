# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /code

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 50505

ENTRYPOINT ["gunicorn", "main:app"]