# Define the specific image the docker image is going to rely on
FROM python:3.8-slim-buster

#  Label Docker file
LABEL Enrico-Ciraci "eciraci@uci.edu"

USER root
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python-all-dev \
    libpq-dev \
    libgeos-dev \
    wget \
    curl \
    sqlite3 \
    cmake \
    libtiff-dev \
    libsqlite3-dev \
    libcurl4-openssl-dev \
    pkg-config


#  Installing PROJ from source
RUN curl https://download.osgeo.org/proj/proj-8.2.1.tar.gz | tar -xz &&\
    cd proj-8.2.1 &&\
    mkdir build &&\
    cd build && \
    cmake .. &&\
    make && \
    make install

#  Installing GDAL from source
RUN wget http://download.osgeo.org/gdal/3.4.0/gdal-3.4.0.tar.gz
RUN tar xvfz gdal-3.4.0.tar.gz
WORKDIR ./gdal-3.4.0
RUN ./configure --with-python --with-pg --with-geos &&\
    make && \
    make install && \
    ldconfig


# set the working directory
WORKDIR /app

# install dependencies
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy the scripts to the folder
COPY . /app

# run test with pytset
CMD ["python", "-m", "pytest", "--import-mode=append", "tests/"]