FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /usr/local/src

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        gnupg \
        environment-modules \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        wget

RUN wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/jammy/amdgpu-install_6.4.60401-1_all.deb
RUN apt install -y /usr/local/src/amdgpu-install_6.4.60401-1_all.deb
RUN apt update -y
RUN amdgpu-install -y --accept-eula --usecase=rocm
RUN usermod -a -G render $LOGNAME
RUN echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
RUN echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf
RUN echo 'EXTRA_GROUPS=render' | sudo tee -a /etc/adduser.conf 


RUN pip3 uninstall tensorflow-rocm
RUN pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/tensorflow_rocm-2.18.1-cp310-cp310-manylinux_2_28_x86_64.whl
RUN pip3 install pandas matplotlib
