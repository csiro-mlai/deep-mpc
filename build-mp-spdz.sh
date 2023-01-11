#!/bin/bash

cd MP-SPDZ || exit 1
echo CXX = clang++ >> CONFIG.mine
echo MY_CFLAGS += -DCHOP_MEMORY >> CONFIG.mine
make mpir
mkdir static
make -j8 emulate.x
