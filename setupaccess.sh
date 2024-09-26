#!/bin/bash

touch group17.pub
echo "$GROUP17_PUBLICKKEY" > group17.pub
echo "setupaccess.sh: make group17.pub file"

touch group17
echo "$GROUP17_PRIVATEKEY" > group17
echo "setupaccess.sh: make group17 file"

chmod 600 group17
echo "setupaccess.sh: change permissions of group17 file"

ssh-keygen -R "[paffenroth-23.dyn.wpi.edu]:22017"
echo "setupaccess.sh: remove known host keys for the server to avoid the REMOTE HOST IDENTIFICATION HAS CHANGED error"

cat group17.pub > authorized_keys
echo "setupaccess.sh: make an authorized_keys file with group17.pub as an authorized key"

scp -o StrictHostKeyChecking=no -i group17 -P 22017 authorized_keys student-admin@paffenroth-23.dyn.wpi.edu:/home/student-admin/.ssh
echo "setupaccess.sh: copy authorized_keys file to server"

rm authorized_keys
echo "setupaccess.sh: remove authorized_keys file from host"

ssh -p 22017 -i group17 -o StrictHostKeyChecking=no student-admin@paffenroth-23.dyn.wpi.edu
echo "setupaccess.sh: try to ssh in"