FROM pytorch/pytorch:latest

COPY . /src

WORKDIR /src

RUN chmod 777 /src/celery-entrypoint.sh

ENTRYPOINT ["/src/celery-entrypoint.sh"]
