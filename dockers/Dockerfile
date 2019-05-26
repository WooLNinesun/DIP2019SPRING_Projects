FROM python:3.7.3-slim

# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Configure apt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install --no-install-recommends apt-utils 2>&1

# Setup python language
RUN apt-get -y install --no-install-recommends locales \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# opencv lib
RUN apt-get -y install --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext-dev

# Clean up
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

# Install jupyter
RUN python -m pip install jupyterlab && jupyter lab --generate-config
COPY dockers/jupyter_config.base.py /root/.jupyter/jupyter_notebook_config.py

# Install Python dependencies from requirements.txt
COPY dockers/requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt && rm requirements.txt

WORKDIR /usr/src

CMD jupyter lab --allow-root
