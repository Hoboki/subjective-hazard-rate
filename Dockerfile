FROM python:3.11

RUN apt-get -y update && apt-get -y install git vim
RUN pip install ipykernel numpy scipy pandas matplotlib h5py torch
