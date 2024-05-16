FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive" TZ="Asia/Tokyo"

RUN apt update && apt install --yes software-properties-common && add-apt-repository --yes ppa:deadsnakes/ppa && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install --yes python3.10 python3.10-dev apt-utils python3-distutils libgl1 wget curl libglib2.0-0 locales && rm -rf /var/lib/apt/lists/*

RUN rm /usr/bin/python3.8 && ln -sf /usr/bin/python3.10 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Add the scripts directory to the PATH to allow easy execution
ENV PATH="/scripts:${PATH}"

# environment setting
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir -U pip

RUN pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /workspace/requirements.txt

# Install required packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

WORKDIR /workspace
