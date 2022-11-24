FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    ffmpeg \
    python3 \
    python3-pip \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools

RUN pip3 install \
    opencv-python \
    tqdm \
    numpy \
    pandas \
    matplotlib

# add user and his password
ARG USER=docker_nclt
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}

RUN mkdir Dataset && chown -R ${UID}:${GID} /home/${USER}
RUN mkdir nclt_dataset_processing && chown -R ${UID}:${GID} /home/${USER}

USER ${UID}:${GID}