FROM python:3.10

RUN apt-get update && apt install git -y --no-install-recommends

COPY . /app

WORKDIR /app
