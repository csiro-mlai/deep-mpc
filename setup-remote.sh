#!/bin/bash

echo "enter username (default ec2-user)"
user=$(read)
sed -i s/root/${user:=ec2-user}/ .ssh/config

echo paste private RSA key followed by Ctrl-d
cat > .ssh/id_rsa
chmod 600 .ssh/id_rsa

echo paste hostnames followed by Ctrl-d
cat > HOSTS
