#!/bin/bash

cd MP-SPDZ
sed -i s#git://github.com#https://github.com#g Makefile .gitmodules
echo CXX = clang++ >> CONFIG.mine
make -j8 tldr
mkdir static
make -j8 {static/,}{{replicated,sy-rep,rep4}-ring-party,emulate}.x
