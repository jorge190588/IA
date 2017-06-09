FROM tensorflow/tensorflow:0.10.0-gpu

RUN apt-get update
RUN apt-get install -y git
RUN pip install pandas
RUN pip install plotly
RUN pip install tflearn
RUN pip install asq

