FROM python:3.9-slim

RUN pip install pipenv
WORKDIR /usr/src/app

COPY . .
RUN pipenv install --deploy
CMD ["pipenv", "run", "flask", "run"]
