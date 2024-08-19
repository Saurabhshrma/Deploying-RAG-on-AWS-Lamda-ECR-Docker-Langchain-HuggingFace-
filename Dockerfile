FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app/

RUN RUN pip install --no-cache-dir *.whl -r requirements.txt

CMD ["streamlit","run","app.py"]
