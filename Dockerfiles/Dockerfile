FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN apt-get update

RUN apt-get -y update && \
    apt-get -y install cmake libopenmpi-dev python3-dev zlib1g-dev

ADD get-pip.py .

ADD requirements.txt .

RUN python get-pip.py

RUN pip install -r requirements.txt

RUN apt-get update

RUN apt-get -y --no-install-recommends install software-properties-common curl apt-transport-https


