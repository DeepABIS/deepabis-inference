FROM tensorflow/tensorflow:1.9.0-py3

RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3-pip \
    libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip

WORKDIR /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "server.py"]

EXPOSE 9042