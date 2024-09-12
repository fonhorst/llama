FROM pytorch/pytorch:latest

COPY requirements.txt /

RUN pip install -r /requirements.txt

RUN pip install tqdm

COPY . /src

WORKDIR /src

ENTRYPOINT ["/src/run.sh"]
