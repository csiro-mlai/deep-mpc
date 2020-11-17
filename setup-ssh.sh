#!/bin/bash

ssh-keygen -f .ssh/id_dsa -N ''
cat .ssh/id_dsa.pub > .ssh/authorized_keys
service ssh start
ssh 127.0.0.1 true
