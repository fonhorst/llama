FROM pytorch/pytorch:latest

COPY requirements.txt /

RUN pip install -r /requirements.txt

RUN pip install "celery[redis]" pydantic "pydantic-settings"

COPY . /src

WORKDIR /src

RUN chmod 777 /src/celery-entrypoint.sh

ENTRYPOINT ["/src/celery-entrypoint.sh"]
