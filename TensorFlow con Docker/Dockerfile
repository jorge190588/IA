FROM tensorflow/tensorflow:0.10.0-gpu
MAINTAINER Jorge Santos

RUN apt-get update 
RUN apt-get install -y git
RUN apt-get install python-matplotlib -y 
RUN pip install pandas
RUN pip install plotly
RUN pip install tflearn
RUN pip install asq
#RUN tensorboard --logdir=run1:/tmp/tensorflow_logs/neurona/ --port 6006

EXPOSE 6006 
EXPOSE 8886 
EXPOSE 8888
ENTRYPOINT /bin/bash
