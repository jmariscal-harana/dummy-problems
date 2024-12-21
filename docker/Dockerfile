# GET BASE IMAGE
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# INSTALL MAIN DEPENDENCIES
RUN apt-get update &&\
    apt-get -y install sudo ffmpeg libsm6 libxext6 x11-apps git curl

# SET UP UBUNTU USER WITH SUDO RIGHTS
RUN adduser ubuntu --disabled-password
RUN sudo usermod -aG sudo ubuntu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

# INSTALL POETRY
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# COPY REPO AND SET UP FOLDERS
WORKDIR /home/ubuntu/dummy-problems
ENV PYTHONPATH="/home/ubuntu/dummy-problems"
ADD . /home/ubuntu/dummy-problems
RUN mkdir -p /home/ubuntu/data && chown -R ubuntu:ubuntu /home/ubuntu/data

# INSTALL PYTHON DEPENDENCIES
RUN poetry export -f requirements.txt --output requirements.txt --dev &&\
    pip install -r requirements.txt