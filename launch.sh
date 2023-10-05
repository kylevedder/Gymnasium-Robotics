#!/bin/bash
xhost +
touch `pwd`/docker_history.txt
mkdir -p `pwd`/pretrained_weights/vip
docker run --gpus=1 --rm -it \
 -v `pwd`:/project \
 -v `pwd`/docker_history.txt:/root/.bash_history \
 -v `pwd`/pretrained_weights/vip:/root/.vip/ \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -e DISPLAY=$DISPLAY \
 -h $HOSTNAME \
 --privileged \
 gymnasium_robotics_playground:latest