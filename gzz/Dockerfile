# Use an official Python runtime as a parent image
FROM python:3.9.16

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install required packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the local code to the container image
COPY . /app

# Set the working directory in the container
WORKDIR /app
