# GET BASE IMAGE
FROM nvcr.io/nvidia/pytorch:24.12-py3

# INSTALL MAIN DEPENDENCIES
RUN apt-get update &&\
    apt-get -y install sudo ffmpeg libsm6 libxext6 x11-apps curl

# DISABLE PASSWORD FOR UBUNTU USER
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

# INSTALL POETRY
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# COPY REPO AND SET UP FOLDERS
WORKDIR /home/ubuntu/dummy-problems
ENV PYTHONPATH="/home/ubuntu/dummy-problems"
ADD --chown=ubuntu:ubuntu . /home/ubuntu/dummy-problems
RUN mkdir -p /home/ubuntu/data && chown -R ubuntu:ubuntu /home/ubuntu/data

# INSTALL PYTHON DEPENDENCIES
RUN pip install lightning timm seaborn gym pygame