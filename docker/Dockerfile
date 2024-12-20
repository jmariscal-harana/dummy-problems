FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update &&\
    apt-get -y install sudo ffmpeg libsm6 libxext6 x11-apps git

RUN pip install pytest opencv-python opencv-contrib-python ipykernel matplotlib

RUN adduser ubuntu --disabled-password
RUN sudo usermod -aG sudo ubuntu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ubuntu
WORKDIR /home/ubuntu
COPY . dummy-problems

WORKDIR /home/ubuntu/dummy-problems

RUN mkdir -p /home/ubuntu/data && chown -R ubuntu:ubuntu /home/ubuntu/data

ENV PYTHONPATH="${PYTHONPATH}:/home/ubuntu/dummy-problems/"