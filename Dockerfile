FROM tensorflow/tensorflow:nightly-gpu

COPY . /app
WORKDIR /app

CMD ["python", "ultrafiltro.py"]
