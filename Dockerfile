FROM sinzlab/pytorch:v3.9-torch1.13.1-cuda11.7.0-dj0.12.9
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade six


RUN pip install git+https://github.com/sinzlab/nnsysident.git@iclr2023
RUN pip install git+https://github.com/sinzlab/neuralpredictors.git@iclr2023

RUN pip3 --no-cache-dir install wandb

RUN pip install --upgrade scikit-image
RUN pip install --upgrade numpy==1.23.5

# install the current project
WORKDIR /project
RUN mkdir /project/neuralmetrics
COPY ./neuralmetrics /project/neuralmetrics
COPY ./setup.py /project
COPY ./pyproject.toml /project

RUN python -m pip install -e /project
