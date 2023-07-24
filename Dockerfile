FROM pytorch/pytorch:latest

COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY . /src

WORKDIR /src

ENTRYPOINT ["/src/run.sh"]
