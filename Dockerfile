FROM tensorflow/tensorflow:1.0.0

RUN pip install plotly
WORKDIR /notebooks
CMD /run_jupyter.sh