FROM node:10.15.3-slim AS node

FROM python:3.7.3-slim

COPY --from=node /usr /usr

# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Configure apt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install --no-install-recommends apt-utils 2>&1

RUN apt-get update && apt-get -y install --no-install-recommends \
    # for python language
    locales \
    # for opencv lib
    libglib2.0-0 libsm6 libxrender1 libxext-dev

# Clean up
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

# Setup python language
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Install jupyter
RUN python -m pip install jupyterlab && jupyter lab --generate-config
COPY dockers/jupyter_config.base.py /root/.jupyter/jupyter_notebook_config.py

# Install jupyter extension
RUN \
    # https://github.com/ryantam626/jupyterlab_code_formatter
    jupyter labextension install @ryantam626/jupyterlab_code_formatter \
    && pip install jupyterlab_code_formatter autopep8 \
    && jupyter serverextension enable --py jupyterlab_code_formatter

# Install Python dependencies from requirements.txt
COPY dockers/requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt && rm requirements.txt

WORKDIR /usr/src

CMD jupyter lab --allow-root
