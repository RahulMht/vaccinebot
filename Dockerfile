FROM python:3.11.0

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./data  /code/data
COPY ./chainlit.md /code/chainlit.md
COPY ./.env /code/.env
COPY ./app.py /code/app.py
COPY ./.chainlit /code/.chainlit

CMD ["chainlit", "run","-w","app.py", "--port", "8080"]


