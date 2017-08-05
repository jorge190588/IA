FROM tensorflow/tensorflow:0.10.0-gpu
MAINTAINER Jorge Santos

RUN apt-get update 
RUN apt-get install -y git

RUN pip install pandas
RUN pip install plotly
RUN pip install tflearn
RUN pip install asq

EXPOSE 6006 
EXPOSE 8886 
EXPOSE 8888
ENTRYPOINT /bin/bash

#RUN IMAGE
# docker build -t tensorflowdemo:1.0 .
#RUN CONTAINER
# docker run --name tensorflowdemo -it tensorflowdemo:1.0 -p 6006:6006 -p 8886:8886 -p 8888:8888

#console
#docker exec -i -t tensorflowdemo bash

#run jupyter notebook on 8888 port
#jupyter notebook

#run tensorflow builder
#tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006


