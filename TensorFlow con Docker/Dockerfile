FROM tensorflow/tensorflow:0.10.0-gpu
MAINTAINER Jorge Santos

RUN apt-get update 
RUN apt-get install -y git
RUN apt-get install wget
RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install plotly
RUN pip install tflearn
RUN pip install asq
# RUN apt-get install python3-matplotlib
# RUN apt-get install python-matplotlib
RUN pip install tensorflow --upgrade

EXPOSE 6006 
EXPOSE 8886 
EXPOSE 8888
ENTRYPOINT "jupyter notebook" /bin/bash
