#!/bin/bash

cd MP-SPDZ || exit 1
echo CXX = clang++ >> CONFIG.mine
echo MY_CFLAGS += -DCHOP_MEMORY >> CONFIG.mine
git clone https://github.com/wbhart/mpir
make -j8 tldr
mkdir static
make -j8 {static/,}{{{replicated,sy-rep,rep4}-ring,{t,h}emi,atlas}-party,emulate}.x
