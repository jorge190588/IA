FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Jorge Santos

RUN apt-get update 
RUN apt-get install -y git
RUN apt-get install wget
RUN pip install --upgrade pip
RUN pip install --upgrade python
RUN apt-get --yes --force-yes install python-pip
RUN pip install numpy
RUN pip install scipy
RUN python -m pip install --user numpy scipy
RUN pip install plotly
RUN pip install tflearn
RUN pip install asq
RUN apt-get install python-pandas -y
RUN apt-get update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN pip install -U scikit-learn
RUN apt-get install python-opencv -y
RUN apt-get --yes --force-yes install cuda
# RUN sudo apt-get --yes --force-yes install cuda-runtime-7-0
# RUN sudo apt-get --yes --force-yes install cuda-7-0
# RUN sudo apt-get --yes --force-yes install libcuda1-384  
# RUN sudo apt-get --yes --force-yes install nvidia-361-dev
# RUN sudo apt-get --yes --force-yes install libdc1394-22-dev libdc1394-22 libdc1394-utils
# RUN sudo apt-get --yes --force-yes uninstall pip ipykernel


# RUN apt-get install python3-matplotlib
# RUN apt-get install python-matplotlib
# RUN sudo pip install tensorflow --upgrade

EXPOSE 6006 
EXPOSE 8886 
EXPOSE 8888

CMD jupyter notebook --allow-root