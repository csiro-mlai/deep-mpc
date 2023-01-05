FROM ubuntu:18.04

RUN apt-get update && apt-get -y install wget tar openssl git make cmake \
    python3 python3-pip clang libsodium-dev autoconf automake \
    libtool yasm texinfo libboost-dev libssl-dev libboost-system-dev \
    libboost-thread-dev libgmp-dev rsync ssh openssh-server procps

WORKDIR /root

ADD download.sh .
RUN ./download.sh

RUN pip3 install numpy

ADD prepare.py .
RUN ./prepare.py

RUN git clone https://github.com/data61/MP-SPDZ
RUN cd MP-SPDZ; git checkout a3acf6e8a304ae996dc8624c6315e9d9e0b79107

ADD build-mp-spdz.sh .
RUN ./build-mp-spdz.sh

ADD ssh_config .ssh/config
ADD setup-ssh.sh .
RUN ./setup-ssh.sh

ADD convert.sh *.py ./
RUN ./convert.sh

ADD *.sh *.py HOSTS ./
RUN ./test_protocols.sh

RUN ./run-local.sh emul D prob 2 3 32 adamapprox
RUN service ssh start; ./run-remote.sh sh3 A near 1 1 16 rate.1
