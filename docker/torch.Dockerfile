FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        cmake
RUN apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8


RUN wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip && \
    unzip libtorch-shared-with-deps-latest.zip
